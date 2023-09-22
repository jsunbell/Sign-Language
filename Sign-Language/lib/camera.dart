import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
// import 'package:flutter_mediapipe/flutter_mediapipe.dart';
import 'package:flutter_mediapipe/gen/landmark.pb.dart';
import 'package:sign_language/video.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  bool _isLoading = true;
  bool _isRecording = false;
  late CameraController _cameraController;

  @override
  void initState() {
    _initCamera();
    super.initState();
  }

  // Future<void> _initializeCamera() async {
  //   final cameras = await availableCameras();
  //   final camera = cameras.first;
  //   _cameraController = CameraController(camera, ResolutionPreset.high);
  //   await _cameraController.initialize();
  //   setState(() {
  //     _isLoading = false;
  //   });
  // }

  void _onLandMarkStream(NormalizedLandmarkList landmarkList) {
    landmarkList.landmark.asMap().forEach((int i, NormalizedLandmark value) {
      print('Index: $i \n' + '$value');
    });
  }

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }

  _initCamera() async {
    final cameras = await availableCameras();
    final back = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back);
    _cameraController = CameraController(back, ResolutionPreset.max);
    await _cameraController.initialize();
    setState(() => _isLoading = false);
  }

  _recordVideo() async {
    if (_isRecording) {
      final file = await _cameraController.stopVideoRecording();
      setState(() => _isRecording = false);
      final route = MaterialPageRoute(
        fullscreenDialog: true,
        builder: (_) => VideoPage(filePath: file.path),
      );
      Navigator.push(context, route);
    } else {
      await _cameraController.prepareForVideoRecording();
      await _cameraController.startVideoRecording();
      setState(() => _isRecording = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text('Sign Language'),
        titleTextStyle: TextStyle(color: Colors.black),
        backgroundColor: Colors.white,
      ),
      body: Center(
          child: Stack(
        alignment: Alignment.center,
        children: [
          Align(
            alignment: Alignment.topCenter,
            child: Container(
              clipBehavior: Clip.antiAlias,
              width: 420,
              height: 400,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(47),
                    bottomRight: Radius.circular(47)),
                color: Colors.white,
              ),
              child: CameraPreview(_cameraController),
            ),
          ),
          Container(
            margin: const EdgeInsets.fromLTRB(0, 70, 0, 0),
            padding: const EdgeInsets.all(25),
            child: FloatingActionButton(
              backgroundColor: Colors.white,
              child: Icon(_isRecording ? Icons.stop : Icons.circle,
                  color: Colors.black),
              onPressed: () => _recordVideo(),
            ),
          ),
          Container(
            margin: const EdgeInsets.fromLTRB(0, 400, 0, 0),
            child: Text(
              'Text',
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.black,
                fontSize: 30,
                fontWeight: FontWeight.w300,
              ),
            ),
          ),
        ],
      )),
    );
  }
}
