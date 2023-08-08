import 'dart:convert';
import 'package:chatgpt_at_home/role.dart';
import 'package:eventsource/eventsource.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'chat_bot_app.dart';
// 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

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    final sp = await SharedPreferences.getInstance();
    apiKey = sp.getString(apiKeyId) ?? const String.fromEnvironment(apiKeyId);
  } catch (err, st) {
    debugPrint('failed to retrieve API key: $err');
    debugPrint('Stack trace: $st');
  }
  runApp(const ChatBotApp());
}

Future<SharedPreferences> get sp => SharedPreferences.getInstance();

var apiKey = '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';
const apiKeyId = 'openaiApiKey';
const material3 = bool.fromEnvironment('material3', defaultValue: true);
const user = 'chatgpt@home';
const done = '[DONE]';
Map<String, String> get requestHeaders =>
    {'content-type': 'application/json', 'authorization': 'Bearer $apiKey'};

final completionEndpoint = Uri.https('api.openai.com', '/v1/chat/completions');

void openLinks(String _, String? href, String title) async {
  if (href != null) {
    await launchUrl(
      Uri.parse(href),
      webOnlyWindowName: title,
      mode: LaunchMode.externalApplication,
    );
  }
}

/// see [docs](https://platform.openai.com/docs/api-reference/chat/create)
Stream<String> chat(
  List<ChatMessage> messages, {
  String model = 'gpt-3.5-turbo',
  void Function(dynamic)? onError,
}) async* {
  final request = jsonEncode({
    'model': model,
    'stream': true,
    'user': user,
    'messages': [
      for (final message in messages)
        if (message.role != Role.system)
          {'role': message.role.api, 'content': message.text}
    ]
  });
  final events = await EventSource.connect(
    completionEndpoint,
    method: 'POST',
    headers: requestHeaders,
    body: request,
  ).catchError((error) {
    if (error is EventSourceSubscriptionException) {
      debugPrint('failed to chat: ${error.message}');
    }
    onError?.call(error);

    throw error;
  });
  var role = 'assistant';
  await for (final event in events) {
    if (event.data == done) break;

    final data = jsonDecode(event.data!);
    final Map<String, dynamic> delta = data['choices'][0]['delta'];
    if (delta.containsKey('role')) {
      role = delta['role'];
      debugPrint('talking as: $role');
    } else if (delta.containsKey('content')) {
      yield delta['content'] as String;
    }
  }
}
