import folium
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import sys, os
import json
from datetime import datetime

class MapWindowPassageiros(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mapa de Saídas de Passageiros")
        self.setGeometry(100, 100, 1000, 750)

        # Criar mapa inicial (centrado em Luanda)
        mapa = folium.Map(location=[-8.8383, 13.2344], zoom_start=14)

        # Carregar registos de passageiros
        records_file = "passageiros_registrados.json"
        if os.path.exists(records_file):
            try:
                with open(records_file, 'r', encoding='utf-8') as f:
                    passageiros = json.load(f)
                    
                # Adicionar marcador para cada passageiro que saiu
                for pid, dados in passageiros.items():
                    try:
                        lat = dados.get('latitude', -8.8383)
                        lng = dados.get('longitude', 13.2344)
                        timestamp = dados.get('timestamp', 'Desconhecido')
                        valor = dados.get('valor', 0)
                        
                        # Criar popup com informações
                        popup_text = f"""
                        <b>Passageiro ID: {pid}</b><br>
                        Valor: {valor} kz<br>
                        Hora: {timestamp[:19]}<br>
                        Local: ({lat:.4f}, {lng:.4f})
                        """
                        
                        # Adicionar marcador com ícone customizado
                        folium.Marker(
                            location=[lat, lng],
                            popup=folium.Popup(popup_text, max_width=250),
                            icon=folium.Icon(color='green', icon='users'),
                            tooltip=f"ID {pid}"
                        ).add_to(mapa)
                        
                    except Exception as e:
                        print(f"Erro ao processar passageiro {pid}: {e}")
            except Exception as e:
                print(f"Erro ao carregar registos: {e}")
        else:
            # Mensagem se nenhum registo ainda existe
            folium.Marker(
                location=[-8.8383, 13.2344],
                popup="Nenhum passageiro registado ainda",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(mapa)

        # Adicionar JS para permitir apenas um marcador de táxi ao clicar
        click_js = """
        function addTaxiMarker(e){
            if (typeof window.taxiMarker !== 'undefined') {
                window.map.removeLayer(window.taxiMarker);
            }
            window.taxiMarker = L.marker(e.latlng, {icon: L.icon({
                iconUrl: 'https://cdn-icons-png.flaticon.com/512/61/61168.png',
                iconSize: [32, 32]
            })}).addTo(window.map).bindPopup("Táxi aqui");
        }
        window.map.on('click', addTaxiMarker);
        """
        mapa.get_root().html.add_child(folium.Element(f"<script>{click_js}</script>"))

        # Salvar mapa
        mapa.save("mapa_passageiros.html")

        # Obter caminho absoluto
        file_path = os.path.abspath("mapa_passageiros.html")

        # Criar navegador embutido
        browser = QWebEngineView()
        browser.load(QUrl.fromLocalFile(file_path))

        # Layout: browser + botões
        central = QWidget()
        v = QVBoxLayout()
        central.setLayout(v)

        v.addWidget(browser)

        h = QHBoxLayout()
        btn_save = QPushButton('Salvar Posição')
        btn_load = QPushButton('Carregar Posição')
        btn_close = QPushButton('Fechar')
        h.addWidget(btn_save)
        h.addWidget(btn_load)
        h.addWidget(btn_close)
        v.addLayout(h)

        self.setCentralWidget(central)

        # conectar callbacks
        btn_save.clicked.connect(lambda: self.salvar_posicao(browser))
        btn_load.clicked.connect(lambda: self.carregar_posicao(browser))
        btn_close.clicked.connect(self.close)

    def salvar_posicao(self, browser: QWebEngineView):
        """Executa JS para obter posição do taxiMarker e salva em taxi_position.json"""
        js = "(window.taxiMarker) ? window.taxiMarker.getLatLng().lat + ',' + window.taxiMarker.getLatLng().lng : null;"
        def callback(result):
            if result is None:
                print('[DEBUG] Nenhum marcador de táxi definido no mapa')
                return
            try:
                lat_str, lng_str = result.split(',')
                lat = float(lat_str)
                lng = float(lng_str)
                data = {'latitude': lat, 'longitude': lng, 'timestamp': datetime.now().isoformat()}
                with open('taxi_position.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f'[DEBUG] Posição do táxi salva: ({lat}, {lng})')
            except Exception as e:
                print(f'[ERRO] Não foi possível salvar posição: {e}')
        browser.page().runJavaScript(js, callback)

    def carregar_posicao(self, browser: QWebEngineView):
        """Carrega taxi_position.json e posiciona o marcador no mapa via JS"""
        if not os.path.exists('taxi_position.json'):
            print('[DEBUG] Nenhum ficheiro taxi_position.json encontrado')
            return
        try:
            with open('taxi_position.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            lat = float(data.get('latitude', -8.8383))
            lng = float(data.get('longitude', 13.2344))
            js = f"if (typeof window.taxiMarker !== 'undefined') {{ window.map.removeLayer(window.taxiMarker); }} window.taxiMarker = L.marker([{lat}, {lng}], {{icon: L.icon({{iconUrl: 'https://cdn-icons-png.flaticon.com/512/61/61168.png', iconSize: [32,32]}})}}).addTo(window.map).bindPopup('Táxi aqui'); window.map.setView([{lat}, {lng}], 16);"
            browser.page().runJavaScript(js)
            print(f'[DEBUG] Posição carregada: ({lat}, {lng})')
        except Exception as e:
            print(f'[ERRO] Não foi possível carregar posição: {e}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapWindowPassageiros()
    window.show()
    sys.exit(app.exec_())
