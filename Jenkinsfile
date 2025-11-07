pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile'
            dir '.'
            additionalBuildArgs '--no-cache'
        }
    }

    stages {
        stage('Check Python Version') {
            steps {
                sh 'python --version'
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                echo "🔧 Installing dependencies..."
                pip install --upgrade pip setuptools wheel
                pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found"
                dvc pull || echo "No DVC remote found"
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh 'python app/train.py'
            }
        }

        stage('Track Metrics') {
            steps {
                echo ' Training complete — metrics logged to MLflow'
            }
        }

        stage('Push to DVC + GitHub') {
            steps {
                sh '''
                echo "📤 Pushing updates to DVC and GitHub..."
                dvc add data/ || echo "No data folder"
                dvc push || echo "No DVC remote"
                git config --global user.email "jenkins@local"
                git config --global user.name "Jenkins"
                git add .
                git commit -m "Auto commit" || echo "Nothing to commit"
                git push origin main || echo "No push"
                '''
            }
        }
    }
}
