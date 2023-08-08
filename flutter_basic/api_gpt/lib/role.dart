import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

import 'main.dart';

enum Role {
  user('user'),
  assistant('assistant'),
  system('system');

  final String api;
  const Role(this.api);
}

class ChatMessage extends StatelessWidget {
  const ChatMessage({
    super.key,
    required this.text,
    required this.role,
  });

  final String text;
  final Role role;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    Widget body = MarkdownBody(
      data: text,
      selectable: true,
      onTapLink: openLinks,
      styleSheet: MarkdownStyleSheet.fromTheme(theme).copyWith(
        blockquote: theme.textTheme.bodyMedium!.copyWith(
          color: theme.colorScheme.onSurface,
        ),
        blockquoteDecoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border.all(
            color: theme.colorScheme.secondaryContainer,
          ),
        ),
      ),
    );
    final isUserMessage = role == Role.user;
    if (role == Role.system) {
      body = ListTile(
        leading: const Icon(Icons.warning),
        tileColor: theme.colorScheme.errorContainer,
        title: body,
      );
    }
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 10.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          if (role != Role.system)
            Container(
              margin: const EdgeInsets.only(right: 16.0),
              child: CircleAvatar(
                child: Text(isUserMessage ? 'U' : 'C'),
              ),
            ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                if (role != Role.system)
                  Text(
                    isUserMessage ? 'You' : 'ChatGPT',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                Container(margin: const EdgeInsets.only(top: 5.0), child: body),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
