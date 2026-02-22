// main.dart
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

void main() {
  runApp(PassageirosApp());
}

class PassageirosApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Contador de Passageiros',
      theme: ThemeData.dark(),
      home: ContadorPage(),
    );
  }
}

class ContadorPage extends StatefulWidget {
  @override
  _ContadorPageState createState() => _ContadorPageState();
}

class _ContadorPageState extends State<ContadorPage> {
  String serverUrl = ''; // Link do servidor
  int totalPessoas = 0;
  Timer? timer;

  final TextEditingController _controller = TextEditingController();

  void startFetching() {
    if (timer != null) timer!.cancel();

    timer = Timer.periodic(Duration(seconds: 1), (Timer t) async {
      if (serverUrl.isEmpty) return;

      try {
        final response = await http.get(Uri.parse('$serverUrl/pessoas'));
        if (response.statusCode == 200) {
          final data = json.decode(response.body);
          setState(() {
            totalPessoas = data['total'] ?? 0;
          });
        } else {
          print("Erro ao conectar ao servidor: ${response.statusCode}");
        }
      } catch (e) {
        print("Falha na requisição: $e");
      }
    });
  }

  void stopFetching() {
    if (timer != null) {
      timer!.cancel();
      timer = null;
    }
  }

  @override
  void dispose() {
    stopFetching();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.red, // Fundo vermelho
      appBar: AppBar(
        title: Text('Contador de Passageiros'),
        backgroundColor: Colors.red.shade900,
      ),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _controller,
              decoration: InputDecoration(
                labelText: 'IP ou link do servidor',
                fillColor: Colors.white,
                filled: true,
              ),
              onSubmitted: (value) {
                serverUrl = value;
                startFetching();
              },
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                serverUrl = _controller.text.trim();
                if (serverUrl.isNotEmpty) {
                  startFetching();
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.black,
              ),
              child: Text('Conectar'),
            ),
            SizedBox(height: 50),
            Text(
              '$totalPessoas ${totalPessoas == 1 ? "pessoa" : "pessoas"}',
              style: TextStyle(
                fontSize: 80,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
