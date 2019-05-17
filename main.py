from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

from pre import predict
from face_align import detect

@app.route("/", methods=['GET', 'POST'])
def upload_file():
	if request.method == 'GET':
		return render_template('index.html')
	if request.method == 'POST':
		if 'file' not in request.files:
			print("file not upload")
			return
		file = request.files['file']
		image = file.read()
		face = detect(image)
		if face is not None:
			cv2.imwrite('./img/test.png', face)
			pred_class, probs = predict(image='./img/test.png')
			return render_template('result.html', pred_class = pred_class, probs = probs)
		else:
			pred_class = "face cannot be determined"
			probs = None
			return render_template('result.html', pred_class = pred_class, probs = probs)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")