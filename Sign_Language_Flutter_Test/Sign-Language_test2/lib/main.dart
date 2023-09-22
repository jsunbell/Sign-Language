// import 'package:flutter/material.dart';
// import 'package:sign_language/camera.dart';
// // import 'package:webview_flutter/webview_flutter.dart';

// void main() {
//   WidgetsFlutterBinding.ensureInitialized();
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({Key? key}) : super(key: key);

//   @override
//   Widget build(BuildContext context) {
//     return const MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: WebViewApp(),
//     );
//   }
// }
import 'package:flutter/material.dart';
// import 'package:permission_handler/permission_handler.dart';

import 'camera.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 퍼미션 요청 및 에러 처리
  //await Permission.camera.isDenied;
  //await Permission.camera.request();

  // if (await Permission.microphone.isDenied) {
  //   await Permission.microphone.request();
  // }

  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: WebViewApp(),
    );
  }
}
