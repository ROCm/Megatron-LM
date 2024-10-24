import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput


def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME" || true
        lsb_release -sd || true
        uname -r || true
        cat /sys/module/amdgpu/version || true
        ls /opt/ -la || true
    """
}

// def clean_up_docker() {
//     sh 'docker ps -a || true' // "|| true" suppresses errors
//     sh 'docker kill $(docker ps -q) || true'
//     sh 'docker rm $(docker ps -a -q) || true'
//     sh 'docker rmi $(docker images -q) || true'
//     sh 'docker system prune -af --volumes || true'
// }

// def clean_up_docker_container() {
//     sh 'docker ps -a || true' // "|| true" suppresses errors
//     sh 'docker kill $(docker ps -q) || true'
// }

// //makes sure multiple builds are not triggered for branch indexing
// def resetbuild() {
//     if(currentBuild.getBuildCauses().toString().contains('BranchIndexingCause')) {
//         def milestonesList = []
//         def build = currentBuild

//         while(build != null) {
//             if(build.getBuildCauses().toString().contains('BranchIndexingCause')) {
//                 milestonesList.add(0, build.number)
//             }
//             build = build.previousBuildInProgress
//         }

//         for (buildNum in milestonesList) {
//             milestone(buildNum)
//         }
//     }
// }

// pipeline {
//     agent any

//     stages {
//         stage('Build') {
//             steps {
//                 echo 'Building..'
//             }
//         }
//         stage('Test') {
//             steps {
//                 echo 'Testing..'
//             }
//         }
//         stage('Deploy') {
//             steps {
//                 show_node_info()
//             }
//         }
//     }
// }


pipeline {

    agent any
    // agent {node ('banff-cyxtera-s83-5.ctr.dcgpu')}
    parameters {
        string(name: 'DOCKER_IMAGE', defaultValue: 'megatron-lm:latest', description: 'Docker image name to build')
        string(name: 'CONTAINER_NAME', defaultValue: 'megatron-lm-container', description: 'Docker container name')
        string(name: 'TEST_COMMAND', defaultValue: './run-tests.sh', description: 'Test command to execute in the container')
    }

    environment {
        DOCKER_WORKSPACE = "${env.WORKSPACE}/docker_workspace"
    }

    stages {
       
        stage('Build Docker Image') {
            steps {
                script {
                    // Copy the necessary files into the Docker workspace
                    // sh "cp Dockerfile_amd ${DOCKER_WORKSPACE}/"
                    // dir(DOCKER_WORKSPACE) {
                        // Build Docker image
                    sh "docker build  -f Dockerfile_amd -t ${params.DOCKER_IMAGE} ."
                    }
                }
            }

        stage('Run Docker Container') {
            steps {
                script {
                    // Run the Docker container with the specified name
                    sh "docker run -d --name ${params.CONTAINER_NAME} ${params.DOCKER_IMAGE}"
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    // Execute test command in the running container
                    sh "docker exec ${params.CONTAINER_NAME} ${params.TEST_COMMAND}"
                }
            }
        }

         stage('Cleanup') {
            steps {
                script {
                    // Execute test command in the running container
                    sh "docker stop ${params.CONTAINER_NAME}"
                    sh "docker rm ${params.CONTAINER_NAME}"
                    sh "docker rmi ${params.DOCKER_IMAGE}"
                }
            }
        }

    }
}
