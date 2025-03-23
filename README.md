# Heart Attack Risk Prediction – MLOps Project

Dieses Projekt entwickelt ein ML-gestütztes System zur Einschätzung des individuellen Herzinfarktrisikos.  
Ziel ist es, Ärzt:innen im klinischen Alltag eine fundierte Entscheidungsunterstützung bereitzustellen – auf Basis anonymisierter Gesundheitsdaten, wissenschaftlicher Modelle und transparenter Prognoselogik.

Die Anwendung wurde im Rahmen einer medizinischen Pilotstudie konzipiert und verbindet moderne Machine-Learning-Technologien mit einem robusten, wartbaren MLOps-Stack.

**Live-Demo (Frontend auf Azure):** [Web-Anwendung starten](http://predictmyheart.westeurope.cloudapp.azure.com:8501/)

**Technische Dokumentation & Entwicklerinfos:** [Wiki ansehen](https://github.com/okayh14/MLOps-Project/wiki)  

**Code Coverage:** [![codecov](https://codecov.io/gh/michellebinder/heart-disease-prediction/branch/testing/graph/badge.svg)](https://codecov.io/gh/michellebinder/heart-disease-prediction)

---

# Getting Started

## Variante 1: Docker (empfohlen)

Die bevorzugte Methode für lokale Ausführung und Entwicklung.
Docker bündelt alle Services in einer Umgebung – ohne manuelles Setup oder Abhängigkeitskonflikte.

**Schritt 1:** Docker installieren
Download [Docker Desktop](https://www.docker.com/get-started/)

**Schritt 2:** Repository klonen
```bash
git clone https://github.com/okayh14/MLOps-Project.git
cd MLOps-Project
```

**Schritt 3:** Im Projektverzeichnis folgenden Befehl ausführen

```bash
docker-compose up --build
```

**Wichtig:** 

Der erste Build und Compose-Vorgang kann – abhängig von der Rechenleistung des Computers – zwischen 6 und 15 Minuten in Anspruch nehmen, da dabei ein Training angestoßen wird. Nach dem initialen Build werden alle relevanten Daten über Volumes gespeichert.

Das laufende Training erkennt man daran, dass im Terminal entsprechende Warnmeldungen ausgegeben werden. Diese Meldungen sind kein Grund zur Sorge, sondern zeigen lediglich an, dass das Training aktiv ist. Sobald diese Warnungen aufhören, gibt der Model Service eine Bestätigung aus, dass fünf Modelle erfolgreich geloggt wurden.

Das System ist einsatzbereit, sobald im Terminal folgende Meldung erscheint:

```bash
orchestrator INFO: Application startup complete.
```

Ab diesem Zeitpunkt senden der Model Service und der Data Service im Sekundentakt GET-Requests (Status 200), was als Indikator dient, dass das System aktiv läuft.

Die Web-Oberfläche ist anschließend unter [http://localhost:8501](http://localhost:8501) erreichbar.

Nach dem einmaligen Build kann das System jederzeit über docker-compose up bzw. docker-compose down gestartet oder gestoppt werden. Sämtliche Daten bleiben dank der Nutzung von Volumes dauerhaft erhalten.

**Folgende Services laufen im Hintergrund:**
- Data Service (Port 8001)
- Model Training Service (Port 8002)
- Orchestrator (Port 8003)
- PostgreSQL-Datenbank
  
---

## Variante 2: Manuelles Setup

Falls du das Projekt **ohne Docker** starten möchtest – z. B. zur gezielten Anpassung einzelner Services oder zu Debugging-Zwecken – folge dieser Anleitung:

### Voraussetzungen
- Python 3.9 oder höher
- Git
- `pip` Paketmanager

**Schritt 1:** Repository klonen
```bash
git clone https://github.com/okayh14/MLOps-Project.git
cd MLOps-Project
```

**Schritt 2:** Virtuelle Umgebung erstellen & aktivieren (empfohlen)
```bash
python3 -m venv env
source env/bin/activate    # macOS/Linux
env\Scripts\activate       # Windows
```

**Schritt 3:** Abhängigkeiten installieren
```bash
pip install --upgrade pip
pip install -r backend/data_service/requirements.txt
pip install -r backend/model_training/requirements.txt
pip install -r backend/Orchestrator/requirements.txt
pip install -r test/requirements.txt
```

**Schritt 4:** Tests ausführen (optional)

```bash
pytest test/
```

## How to Contribute
Alle Infos zur lokalen Entwicklung, Branching-Strategie, Testabdeckung, API-Erweiterung und mehr finden sich im
[Kapitel 8 – How to Contribute (Wiki)](https://github.com/okayh14/MLOps-Project/wiki/8.-How-to-Contribute)
