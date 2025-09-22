pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'iris-train'
        CONTAINER_NAME = 'iris-container'
        DOCKER_IMAGE_TAG = "${DOCKER_IMAGE}:${BUILD_NUMBER}"
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
                    // Vérifier si Docker est installé
                    sh 'docker --version'
                    
                    // Construire l'image Docker
                    sh """
                        docker build -t ${DOCKER_IMAGE_TAG} .
                        echo "Image built: ${DOCKER_IMAGE_TAG}"
                    """
                }
            }
        }
        
        stage('Run Training') {
            steps {
                echo '🤖 Starting ML training...'
                script {
                    // Créer le dossier artifacts sur l'hôte
                    sh 'mkdir -p artifacts'
                    
                    // Exécuter le container avec volume mount
                    sh """
                        docker run --name ${CONTAINER_NAME}-${BUILD_NUMBER} \
                        --rm \
                        -v "${WORKSPACE}/artifacts:/app/artifacts" \
                        ${DOCKER_IMAGE_TAG}
                    """
                }
            }
        }
        
        stage('Verify Artifacts') {
            steps {
                echo '🔍 Verifying generated artifacts...'
                script {
                    sh '''
                        echo "Contents of artifacts directory:"
                        ls -la artifacts/
                        
                        if [ -f "artifacts/iris_model.pkl" ]; then
                            echo "✅ Model file found!"
                            file artifacts/iris_model.pkl
                        else
                            echo "❌ Model file not found!"
                            exit 1
                        fi
                    '''
                }
            }
        }
    }
    
    post {
        always {
            echo '🧹 Cleaning up Docker resources...'
            script {
                // Nettoyage sécurisé - ignorer les erreurs si les ressources n'existent pas
                sh """
                    # Supprimer l'image construite
                    docker rmi ${DOCKER_IMAGE_TAG} || echo "Image already removed"
                    
                    # Nettoyer les containers arrêtés
                    docker container prune -f || echo "No containers to clean"
                    
                    # Nettoyer les images non utilisées
                    docker image prune -f || echo "No images to clean"
                """
            }
        }
        
        success {
            echo '✅ Pipeline completed successfully!'
            
            // Archiver les artifacts
            archiveArtifacts artifacts: 'artifacts/**/*', allowEmptyArchive: false
            
            // Afficher un résumé
            script {
                sh '''
                    echo "=== BUILD SUMMARY ==="
                    echo "✅ Model trained successfully"
                    echo "✅ Artifacts generated:"
                    ls -la artifacts/
                    echo "====================="
                '''
            }
        }
        
        failure {
            echo '❌ Pipeline failed!'
            
            // Debug info en cas d'échec
            script {
                sh '''
                    echo "=== DEBUG INFORMATION ==="
                    echo "Docker version:"
                    docker --version || echo "Docker not available"
                    
                    echo "Workspace contents:"
                    ls -la
                    
                    echo "Artifacts directory:"
                    ls -la artifacts/ || echo "Artifacts directory not found"
                    
                    echo "Docker images:"
                    docker images | grep iris-train || echo "No iris-train images found"
                    
                    echo "========================="
                '''
            }
        }
    }
}