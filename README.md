
# Agent Data_Analyst

Extracting rich insights from
the data across organization........

---

##  1. Préparation de l'Environnement Local

Utilisez ces commandes pour initialiser votre espace de travail proprement :

```bash
# 1. Création de l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 2. Création du pyproject.toml 

# 3. Installation des dépendances

```

---

## 2. Configuration du Secret Manager (Tools)

L'agent a besoin d'un fichier de configuration `tools.yaml` stocké de manière sécurisée dans Google Cloud pour accéder à BigQuery.

```bash
# Création secret
gcloud secrets create tools --replication-policy="automatic"

# Ajouter ou mettre à jour la version du fichier tools.yaml
gcloud secrets versions add tools --data-file="tools.yaml"

```

---

## 3. Déploiement du Serveur d'Outils (Cloud Run)

Le serveur MCP (Toolbox) qui contient le tools Conversationnal_Analytics

```bash
# Déploiement sur Cloud Run
gcloud run deploy dataagent \
  --image=us-central1-docker.pkg.dev/database-toolbox/toolbox/toolbox:latest \
  --service-account=bqdatalyzer-toolbox-identity@\${PROJECT_ID}.iam.gserviceaccount.com \
  --set-secrets="/app/tools.yaml=tools:latest" \
  --args="--tools-file=/app/tools.yaml","--address=0.0.0.0","--port=8080" \
  --min-instances=0 \
  --region=us-central1 \
  --project=\${PROJECT_ID} \
  --no-allow-unauthenticated

```

> **Note :** Une fois déployé, copiez l'URL du service et collez-la dans la variable `SERVER_URL` du fichier `BQdatalyzer/agent.py`.

---

## 4. Déploiement de l'Agent (Vertex AI)

Déployez l'orchestrateur (Reasoning Engine) avec le monitoring activé.

```bash
# Déploiement de l'agent avec ADK - Déplacez vous dans le dossier de l'agent /agent/BQdatalyzer
adk deploy agent-engine \
  --project \${PROJECT_ID} \
  --region us-central1 \
  --display_name "dataagent" \
    .

```

---

## 5. Rappel des Permissions (IAM)

Assurez-vous que votre compte de service possède les rôles suivants sur le projet :

* `roles/cloudaicompanion.user` : Accès au chat Gemini.
* `roles/geminidataanalytics.dataAgentStatelessUser` : API Conversational Analytics.
* `roles/bigquery.user` & `roles/bigquery.dataViewer` : Accès aux données sous bigquery.
* `roles/secretmanager.secretAccessor` : Lecture du fichier tools.yaml.

---
