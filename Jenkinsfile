pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/MAliTopkara/jenkins.git'  // veya autogluon_runtime repo URL'in
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                pip install -r requirements.txt
                pip install mlflow==2.14.0 autogluon==1.2.0 dvc[gdrive]==3.53.0
                dvc pull || echo "No DVC remote found, skipping..."
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
                echo 'Model training complete. Logged to MLflow.'
            }
        }

        stage('Push DVC + GitHub') {
            steps {
                sh '''
                dvc add data/ || echo "No data folder to add"
                dvc push || echo "No DVC remote, skipping..."
                git add .
                git commit -m "Auto update model and data via Jenkins" || echo "No changes to commit"
                git push origin main || echo "Git push skipped"
                '''
            }
        }
    }
}
