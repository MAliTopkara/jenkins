pipeline {
    agent any

    stages {
        stage('Setup Environment') {
            steps {
                sh '''
                pip install -r requirements.txt
                pip install mlflow==2.14.0 autogluon==1.2.0 dvc[gdrive]==3.53.0
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
                echo 'Training complete — metrics logged to MLflow'
            }
        }

        stage('Push to DVC + GitHub') {
            steps {
                sh '''
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
