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
                echo 'Cloning repository...'
                checkout scm
            }
        }
        
        stage('Verify Environment') {
            steps {
                echo 'Checking environment...'
                bat '''
                    echo Current directory: %CD%
                    dir
                    echo.
                    echo Docker version:
                    docker --version
                    echo.
                    echo Git version:
                    git --version
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                bat '''
                    echo Building image: %DOCKER_IMAGE_TAG%
                    docker build -t %DOCKER_IMAGE_TAG% .
                    echo.
                    echo Verifying image was built:
                    docker images | findstr %DOCKER_IMAGE%
                '''
            }
        }
        
        stage('Run Training') {
            steps {
                echo 'Starting ML training...'
                bat '''
                    echo Creating artifacts directory...
                    if not exist "artifacts" mkdir artifacts
                    
                    echo Running Docker container for training...
                    docker run --name %CONTAINER_NAME%-%BUILD_NUMBER% --rm -v "%WORKSPACE%\\artifacts:/app/artifacts" %DOCKER_IMAGE_TAG%
                    
                    echo Training completed. Checking artifacts...
                    dir artifacts
                '''
            }
        }
        
        stage('Verify Artifacts') {
            steps {
                echo 'Verifying generated artifacts...'
                bat '''
                    echo Contents of artifacts directory:
                    if exist "artifacts" (
                        dir artifacts
                        echo.
                        if exist "artifacts\\iris_model.pkl" (
                            echo SUCCESS: Model file found!
                            echo File size:
                            for %%I in (artifacts\\iris_model.pkl) do echo %%~zI bytes
                        ) else (
                            echo ERROR: Model file not found!
                            exit 1
                        )
                    ) else (
                        echo ERROR: Artifacts directory not found!
                        exit 1
                    )
                '''
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up Docker resources...'
            bat '''
                echo Removing Docker image: %DOCKER_IMAGE_TAG%
                docker rmi %DOCKER_IMAGE_TAG% 2>nul || echo Image already removed or not found
                
                echo Cleaning up unused Docker resources...
                docker container prune -f 2>nul || echo No containers to clean
                docker image prune -f 2>nul || echo No images to clean
                
                echo Cleanup completed.
            '''
        }
        
        success {
            echo 'Pipeline completed successfully!'
            
            // Archive artifacts
            archiveArtifacts artifacts: 'artifacts/**/*', allowEmptyArchive: false
            
            // Display summary
            bat '''
                echo.
                echo =====================================
                echo        BUILD SUMMARY - SUCCESS
                echo =====================================
                echo Model trained successfully
                echo Artifacts generated:
                dir artifacts
                echo =====================================
            '''
        }
        
        failure {
            echo 'Pipeline failed!'
            
            // Debug information
            bat '''
                echo.
                echo =====================================
                echo         DEBUG INFORMATION
                echo =====================================
                echo Current directory: %CD%
                dir
                echo.
                echo Artifacts directory status:
                if exist "artifacts" (
                    echo Artifacts directory exists:
                    dir artifacts
                ) else (
                    echo Artifacts directory does not exist
                )
                echo.
                echo Docker images with iris-train:
                docker images | findstr iris-train || echo No iris-train images found
                echo.
                echo Docker containers:
                docker ps -a | findstr iris-container || echo No iris-container found
                echo =====================================
            '''
        }
    }
}