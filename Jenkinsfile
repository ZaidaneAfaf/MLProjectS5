pipeline {
    agent any

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build('iris-train')
                }
            }
        }

        stage('Run Training') {
            steps {
                script {
                    docker.image('iris-train').run('--name iris-container')
                }
            }
        }

        stage('Copy Model Artifact') {
            steps {
                sh 'mkdir -p artifacts'
                sh 'docker cp iris-container:/app/iris_model.pkl ./artifacts/iris_model.pkl'
            }
        }

        stage('Clean Up') {
            steps {
                sh 'docker rm iris-container || true'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'artifacts/iris_model.pkl', allowEmptyArchive: false
        }
    }
}
