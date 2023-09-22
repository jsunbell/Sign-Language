import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:permission_handler/permission_handler.dart';

class WebViewApp extends StatefulWidget {
  const WebViewApp({Key? key}) : super(key: key);

  @override
  _WebViewAppState createState() => _WebViewAppState();
}

class _WebViewAppState extends State<WebViewApp> {
  late InAppWebViewController webView;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  // 권한을 요청하는 메소드
  Future<void> _requestPermissions() async {
    // 예: 카메라와 마이크 권한을 요청
    PermissionStatus cameraStatus = await Permission.camera.status;
    PermissionStatus microphoneStatus = await Permission.microphone.status;

    if (cameraStatus.isDenied) {
      await Permission.camera.request();
    }
    if (microphoneStatus.isDenied) {
      await Permission.microphone.request();
    }
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async {
        if (await webView.canGoBack()) {
          webView.goBack();
          return false;
        }
        return true;
      },
      child: Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: const Text('Sign Language'),
          titleTextStyle: TextStyle(color: Colors.black),
          backgroundColor: Colors.white,
        ),
        body: Container(
          child: Column(
            children: <Widget>[
              Expanded(
                child: Container(
                  margin: const EdgeInsets.only(top: 30.0),
                  child: InAppWebView(
                    initialUrlRequest: URLRequest(
                        url: Uri.parse(
                            'https://firebasestorage.googleapis.com/v0/b/seng-q.appspot.com/o/test.html?alt=media&token=88fea095-8e8b-4775-95f6-f930a865cd05')),
                    onWebViewCreated: (InAppWebViewController controller) {
                      webView = controller;
                    },
                    initialOptions: InAppWebViewGroupOptions(
                      crossPlatform: InAppWebViewOptions(
                        useShouldOverrideUrlLoading: true,
                        mediaPlaybackRequiresUserGesture: false,
                        javaScriptEnabled: true,
                      ),
                      android: AndroidInAppWebViewOptions(
                        useHybridComposition: true,
                      ),
                    ),
                    // onWebViewCreated: (InAppWebViewController controller) {
                    //   webView = controller;
                    // webView.loadUrl(
                    //     urlRequest: URLRequest(
                    //         url: Uri.parse('https://www.google.com/'
                    //             // 'https://firebasestorage.googleapis.com/v0/b/seng-q.appspot.com/o/test.html?alt=media&token=88fea095-8e8b-4775-95f6-f930a865cd05'
                    //     )));
                    // },
                    androidOnPermissionRequest:
                        (InAppWebViewController controller, String origin,
                            List<String> resources) async {
                      return PermissionRequestResponse(
                          resources: resources,
                          action: PermissionRequestResponseAction.GRANT);
                    },
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}
