import 'dart:convert';

import 'package:chatgpt_at_home/role.dart';
import 'package:eventsource/eventsource.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart';

import 'main.dart';
import 'searchable_list_view.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _textController = TextEditingController();

  // Define the list of chat messages
  final List<ChatMessage> _messages = [];
  final stt = SpeechToText();
  LocaleName? currentLocale;
  String? interim;
  List<LocaleName> locales = [];

  var sttInitialized = false;

  @override
  void initState() {
    super.initState();

    (() async {
      final initialized =
          await stt.initialize(debugLogging: kDebugMode).catchError((err) {
        debugPrint('Could not initialize speech-to-text: $err');
        return false;
      });

      setState(() => sttInitialized = initialized);
      if (initialized) {
        final locales = await stt.locales();
        setState(() => this.locales = locales);
      }
    })();
  }

  void _onSelectLanguage() async {
    final choice = await showDialog<LocaleName>(
      context: context,
      builder: (context) {
        String? needle;
        Function(Function()) setState = (_) {};

        return Card(
          child: Center(
            child: Column(children: [
              TextFormField(
                decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.search),
                  hintText: 'Search dictation locales',
                ),
                onChanged: (input) {
                  setState(() {
                    needle = input;
                  });
                },
              ),
              Expanded(
                child: StatefulBuilder(builder: (context, $setState) {
                  // HACK: setState escapes builder
                  setState = $setState;
                  return SearchableListView.builder(
                    searching: needle != null,
                    onFilter: (i) => locales[i]
                        .name
                        .toLowerCase()
                        .contains(needle!.toLowerCase()),
                    itemCount: locales.length,
                    itemBuilder: (context, i) => ListTile(
                        title: Text(locales[i].name),
                        onTap: () => Navigator.pop(context, locales[i])),
                  );
                }),
              )
            ]),
          ),
        );
      },
    );
    if (choice != null) {
      setState(() {
        currentLocale = choice;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ChatGPT@Home'), actions: [
        if (locales.isNotEmpty)
          IconButton(
            icon: const Icon(Icons.language),
            onPressed: _onSelectLanguage,
          ),
        IconButton(
          icon: const Icon(Icons.cleaning_services),
          onPressed: _onClearMessages,
        ),
        IconButton(
          icon: const Icon(Icons.settings),
          onPressed: _onOpenSettings,
        )
      ]),
      body: Column(children: [
        Flexible(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: CustomScrollView(
              reverse: true,
              slivers: [
                if (interim != null)
                  SliverToBoxAdapter(
                    child: ChatMessage(
                      text: interim!,
                      role: Role.assistant,
                    ),
                  ),
                SliverList(
                  delegate: SliverChildBuilderDelegate(
                    (_, index) => _messages[_messages.length - index - 1],
                    childCount: _messages.length,
                  ),
                ),
              ],
            ),
          ),
        ),
        const Divider(height: 1.0),
        Container(
          decoration: BoxDecoration(color: Theme.of(context).cardColor),
          child: Builder(builder: _buildTextComposer),
        ),
      ]),
    );
  }

  Widget _buildTextComposer(BuildContext context) {
    final hasApiKey = apiKey.isNotEmpty;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8.0),
      child: Row(
        children: <Widget>[
          if (sttInitialized)
            IconButton(
              icon: const Icon(Icons.mic),
              onPressed: () async {
                var interim = '...';
                Function(Function()) setState = (_) {};
                final ctl = showBottomSheet(
                  context: context,
                  builder: (context) {
                    return StatefulBuilder(builder: (context, $setState) {
                      // HACK: setState escapes the builder.
                      setState = $setState;
                      return Card(
                        child: SizedBox(
                          height: 200,
                          width: double.infinity,
                          child: Center(
                            child: Text(
                              interim,
                              style: Theme.of(context).textTheme.bodyLarge,
                            ),
                          ),
                        ),
                      );
                    });
                  },
                );
                stt.errorListener = (_) => ctl.close();
                await stt.listen(
                  cancelOnError: true,
                  listenMode: ListenMode.dictation,
                  localeId: currentLocale?.localeId,
                  onResult: (result) {
                    if (result.finalResult) {
                      stt.stop();
                      ctl.close();
                      addMessage(ChatMessage(
                        text: result.recognizedWords,
                        role: Role.user,
                      ));
                      return;
                    }

                    setState(() {
                      interim = result.recognizedWords;
                    });
                  },
                );
              },
            ),
          Expanded(
            child: TextField(
              controller: _textController,
              onSubmitted: _handleSubmitted,
              decoration: InputDecoration.collapsed(
                hintText: hasApiKey ? 'Ask me anything' : 'No API key set',
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.send),
            onPressed: !hasApiKey
                ? null
                : () => _handleSubmitted(_textController.text),
          ),
        ],
      ),
    );
  }

  void _handleSubmitted(String text) {
    _textController.clear();
    addMessage(ChatMessage(
      text: text,
      role: Role.user,
    ));
  }

  void addMessage(ChatMessage chatMessage) async {
    if (chatMessage.text.isEmpty) return;

    setState(() {
      _messages.add(chatMessage);
      interim = '';
    });

    void onError(err) {
      var message = err.toString();
      if (err is EventSourceSubscriptionException) {
        message = jsonDecode(err.message)['error']['message'];
      }

      setState(() {
        interim = null;
        _messages.add(ChatMessage(
          text: message,
          role: Role.system,
        ));
      });
    }

    await for (final delta in chat(_messages, onError: onError)) {
      setState(() {
        interim = interim! + delta;
      });
    }

    setState(() {
      _messages.add(ChatMessage(text: interim!, role: Role.assistant));
      interim = null;
    });
  }

  void _onClearMessages() {
    setState(() {
      _messages.clear();
    });
  }

  void _onOpenSettings() async {
    await showDialog(
      context: context,
      builder: (BuildContext context) {
        String newApiKey = apiKey;

        return AlertDialog(
          title: const Text('Set API Key'),
          content: TextFormField(
            initialValue: newApiKey,
            obscureText: true,
            decoration: const InputDecoration(hintText: 'Enter your API key'),
            onChanged: (value) => newApiKey = value,
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('Cancel'),
              onPressed: () {
                Navigator.pop(context);
              },
            ),
            TextButton(
              child: const Text('Save'),
              onPressed: () async {
                await (await sp).setString(apiKeyId, apiKey = newApiKey);
                setState(() {
                  if (mounted) Navigator.pop(context);
                });
              },
            ),
          ],
        );
      },
    );
  }
}
