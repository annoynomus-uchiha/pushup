document.addEventListener('DOMContentLoaded', function() {
    const videoUpload = document.getElementById('videoUpload');
    const startButton = document.getElementById('startProcessing');
    const videoPreview = document.getElementById('videoPreview');
    const canvasOutput = document.getElementById('canvasOutput');
    const resultsDiv = document.getElementById('results');

    let poseDetector;
    let videoStream;
    let pushupCounter = {
        pushups: 0,
        lastState: 'up',
        state: 'up',
        angle: 0,
        ok: 0,
        notOk: 0,
        calcPushup: function (hip, shoulder, ankle) {
            let a = tf.tensor1d(hip);
            let b = tf.tensor1d(shoulder);
            let c = tf.tensor1d(ankle);
            const radians = tf.atan2(tf.sub(c.slice(1,1),b.slice(1,1)), tf.sub(c.slice(0,1),b.slice(0,1)))
                    .sub(tf.atan2(tf.sub(a.slice(1,1),b.slice(1,1)), tf.sub(a.slice(0,1),b.slice(0,1))));
            const angleTensor = tf.abs(radians.mul(180.0/Math.PI));
            const angle = angleTensor.dataSync()[0];
           // This will need to be adjusted based on your testing
            if (angle > 160) {
                this.state = 'down'
            } else if (angle < 100) {
                this.state = 'up';
            }
            if(this.state != this.lastState && this.state == 'up'){
                this.pushups++;
                if(angle > 120 && angle < 160)
                    this.ok++;
                else
                    this.notOk++;
            }
            this.lastState = this.state;
            angleTensor.dispose();
            a.dispose();
            b.dispose();
            c.dispose();
            return angle;
        }
    };

    async function setupPoseDetection() {
      const model = poseDetection.SupportedModels.BlazePose;
        const detectorConfig = {
            runtime: 'tfjs',
            modelType: 'lite',
            maxPoses: 1
        };
        poseDetector = await poseDetection.createDetector(model, detectorConfig);
    }

    async function processVideo() {
        if (!poseDetector) {
          await setupPoseDetection();
        }
        const canvasCtx = canvasOutput.getContext('2d');
        const videoWidth = videoPreview.videoWidth;
        const videoHeight = videoPreview.videoHeight;
        canvasOutput.width = videoWidth;
        canvasOutput.height = videoHeight;
        canvasCtx.clearRect(0, 0, videoWidth, videoHeight);
        async function frameProcessing() {
            if (videoPreview.paused || videoPreview.ended)
                return;
            if (poseDetector) {
                const poses = await poseDetector.estimatePoses(videoPreview);
                if (poses && poses.length > 0) {
                    const pose = poses[0];
                    if (pose && pose.keypoints) {
                            canvasCtx.drawImage(videoPreview, 0, 0, videoWidth, videoHeight);
                            for(let k of pose.keypoints) {
                                canvasCtx.fillStyle = 'red'
                                canvasCtx.fillRect(k.x - 2, k.y-2, 4, 4);
                            }
                        // Get coordinates
                        let shoulder = [pose.keypoints[5].x,pose.keypoints[5].y];
                        let hip = [pose.keypoints[11].x,pose.keypoints[11].y];
                        let ankle = [pose.keypoints[15].x,pose.keypoints[15].y];
                        try {
                           let angle = pushupCounter.calcPushup(hip, shoulder, ankle);
                           canvasCtx.fillStyle = 'white';
                            canvasCtx.font = '16px serif';
                            canvasCtx.fillText(angle.toFixed(2), shoulder[0], shoulder[1]);
                        } catch(e) {
                          console.log(e);
                        }
                         resultsDiv.innerHTML = `<p>Push-ups: ${pushupCounter.pushups}</p>
                                                 <p>OK: ${pushupCounter.ok}</p>
                                                 <p>NOT OK: ${pushupCounter.notOk}</p>`;
                    }
                }
            }
            requestAnimationFrame(frameProcessing);
        }
        requestAnimationFrame(frameProcessing)
    }
    startButton.addEventListener('click', async function() {
        const file = videoUpload.files[0];
        if(!file){
            alert('No File selected');
            return;
        }
        videoPreview.src = URL.createObjectURL(file);
        videoPreview.style.display = 'block'
        videoPreview.play();
        videoPreview.onloadeddata = async () => {
            await processVideo();
        };
    })
});