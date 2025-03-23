Code Coverage: [![codecov](https://codecov.io/gh/michellebinder/heart-disease-prediction/branch/testing/graph/badge.svg)](https://codecov.io/gh/michellebinder/heart-disease-prediction)

Frontend (gehosted bei Azure):  [![Web-Anwendung](http://predictmyheart.westeurope.cloudapp.azure.com:8501/ )]

# MLOps Project
Dieses Projekt umfasst eine End-to-End-MLOps-Lösung, die Datenverarbeitung, Modelltraining, Inferenz und Orchestrierung über FastAPI-Microservices integriert. Das Projekt umfasst CI/CD-Automatisierung mit GitHub Actions und Testabdeckungsberichte über Codecov.

## Overview
Dieses Projekt modelliert eine vollständige MLOps-Pipeline mit modularer Microservices-Architektur:
- **Data Service**: Datenbereinigung und -präprozessierung
- **Model Training**: Training, Inferenz and Modellverwaltung
- **Orchestrator**: Koordinierung der Services und API Management

**Ordnerstruktur**
```
backend/
  ├── data_service/           # Datenvorverarbeitung und -bereinigung
  ├── model_training/         # Training, Inferenz, Modellverwaltung
  ├── Orchestrator/           # FastAPI Serviceorchestrierung
test/
  ├── integration/            # Integration tests
  ├── unit/                   # Unit tests
```
### Quick Navigation:

>Für Funktionalitäten bezüglich Modelltraining und -verwaltung sowie Inferenz: [`backend/model_service`](./backend/model_service)

>Für Funktionalitäten bezüglich Datenmanagement: [`backend/data_service`](./backend/data_service)

>Für Funktionalitäten bezüglich Orchestrierung: [`backend/Orchestrator`](./backend/Orchestrator)

## CI/CD Pipeline
Die CI/CD-Pipeline läuft automatisch bei Push und Pull-Requests und umfasst:

- Code Formatierung mit `Black`
- Linting mit `Flake8`
- Entfernen ungenutzter Importe mit `Autoflake`
- Unit und Integrationtests mit Testcoverage Monitoring
- Testcoverage upload nach `Codecov`

Durch einen Push und durch eine Pull-Request wir die CI/CD-Pipeline gestartet.

## Branching-Strategie

Zur strukturierten Zusammenarbeit im Team folgt dieses Projekt einer klaren Branching-Strategie:

- `main`: Stabiler, getesteter Code. Repräsentiert die aktuelle, funktionierende Version.

- `production`: Wird von `main` abgezweigt und dient für den Deployment auf Azure.

- `develop`: Hauptentwicklungszweig. Hier fließen neue Features und Bugfixes ein.

- `testing`: Dient zur Implementierung und Ausführung von Tests vor dem Merge in `main`.

### Feature- und Bugfix-Branches

**Neues Feature:**
- branch von develop: feature/feature-name

**Bugfix:**
- branch von develop: bugfix/bug-name

Sobald die Entwicklung eines Features oder Bugfixes abgeschlossen ist, wird der entsprechende Branch in den `develop` - Branch gemergt.

Anschließend kann ein temporärer `testing`- Branch von `develop`erstellt werden, um umfassende Tests durchzuführen.

Nach erfolgreichem Abschluss aller Tests und Code-Reviews wird der `develop`- Branch in den `main`- Branch gemergt, wodurch ein stabiler, produktionsreifer Stand sichergestellt wird.
