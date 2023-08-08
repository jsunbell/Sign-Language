import 'package:flutter/material.dart';

import 'chat_screen.dart';
import 'main.dart';

class ChatBotApp extends StatelessWidget {
  const ChatBotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ChatGPT@Home',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.light(useMaterial3: material3),
      darkTheme: ThemeData.dark(useMaterial3: material3),
      home: const ChatScreen(),
    );
  }
}
