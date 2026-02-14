echo "⚙️  Configuration de l'environnement de build..."
pip install "setuptools<70.0.0" wheel

echo "🚀 Installation des dépendances du projet..."
pip install -r requirements.txt

echo "✅ Installation terminée avec succès !"