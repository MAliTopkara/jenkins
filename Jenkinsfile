pipeline {
    agent {
        docker {
            image 'python:3.12-slim'
            args '-v /var/run/docker.sock:/var/run/docker.sock -v $WORKSPACE:/workspace'
        }
    }
    environment {
        MLFLOW_TRACKING_URI = 'http://mlflow:5000'
    }
    stages {
        stage('Setup') {
            steps {
                script {
                    sh 'pip install -r workspace/requirements.txt'
                }
            }
        }
        stage('Run Training') {
            steps {
                script {
                    sh 'python workspace/app/train.py'
                }
            }
        }
        stage('Verify Artifacts') {
            steps {
                script {
                    sh 'ls -la workspace/mlruns'
                }
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'workspace/mlruns/**', allowEmptyArchive: true
        }
        success {
            echo 'Training completed successfully!'
        }
        failure {
            echo 'Training failed!'
        }
    }
}