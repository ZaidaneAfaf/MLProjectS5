pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'iris-train'
        CONTAINER_NAME = 'iris-container'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '📦 Cloning repository...'
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo '🔨 Building Docker image...'
                script {
                    def image = docker.build("${DOCKER_IMAGE}:${BUILD_NUMBER}")
                    env.DOCKER_IMAGE_TAG = "${DOCKER_IMAGE}:${BUILD_NUMBER}"
                }
            }
        }
        
        stage('Run Training') {
            steps {
                echo '🤖 Starting ML training...'
                script {
                    sh """
                        docker run --name ${CONTAINER_NAME}-${BUILD_NUMBER} \
                        --rm -v \$(pwd)/artifacts:/app/artifacts \
                        ${DOCKER_IMAGE_TAG}
                    """
                }
            }
        }
        
        stage('Copy Model Artifact') {
            steps {
                echo '💾 Collecting model artifacts...'
                sh '''
                    mkdir -p artifacts
                    # Le modèle est déjà dans artifacts/ grâce au volume mount
                    ls -la artifacts/
                '''
            }
        }
    }
    
    post {
        always {
            echo '🧹 Cleaning up...'
            sh """
                docker rmi ${DOCKER_IMAGE_TAG} || true
                docker system prune -f
            """
        }
        
        success {
            echo '✅ Pipeline completed successfully!'
            archiveArtifacts artifacts: 'artifacts/iris_model.pkl', allowEmptyArchive: false
            
            // Notifications (optionnel)
            emailext (
                subject: "✅ ML Training Success - Build ${BUILD_NUMBER}",
                body: "Le modèle Iris a été entraîné avec succès!",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
        
        failure {
            echo '❌ Pipeline failed!'
            emailext (
                subject: "❌ ML Training Failed - Build ${BUILD_NUMBER}",
                body: "L'entraînement du modèle a échoué. Vérifiez les logs.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}