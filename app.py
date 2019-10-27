# -*- coding: UTF-8 -*-
from flask import Flask, request, redirect, url_for, render_template
from flask import send_from_directory
from base64 import b64encode
from io import BytesIO
import image_process as image_process


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg','JPG'])


@app.route('/')
def index():
    return render_template('camera.html')


@app.route('/post', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_url_1 = request.form['image']
        img_url = img_url_1.split(",")[1]
        #ここで画像を渡す
        # new_img_url = image_process.main(img_url)
        new_img_url = "data:image/png;base64,{}".format(img_url) 
        return render_template('result.html', img_url=new_img_url)

if __name__ == '__main__':
    app.debug = True
    # app.run(host='0.0.0.0', ssl_context=('open_ssl/server.crt', 'open_ssl/server.key'), threaded=True, debug=True)
    app.run(host='0.0.0.0', ssl_context=('open_ssl/server.crt', 'open_ssl/server.key'))
