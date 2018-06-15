import os,json
from flask import Flask, request, url_for, send_from_directory,redirect,render_template,jsonify
from werkzeug import security
from werkzeug.datastructures import ImmutableMultiDict
from AI import bot

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

app = Flask(__name__,static_url_path='')
app.config['UPLOAD_FOLDER'] = 'temp/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

global ai
ai = bot()


html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>图片上传</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    '''
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/uploads/<dealedfile>')
def dealed_file(dealedfile):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               dealedfile)

@app.route('/show')
def show_url():
    return app.send_static_file('index1.html')


@app.route('/dealed')
def dealed_url():
    # 把你的处理方法写到这里，记住，上传的文件名必须是1.jpg,经过处理后的文件名必须是2.jpg
    result = ai.predict('D:/Medical-image-segment-master/temp/1.jpg')
    dict = {"result": "1"}
    return json.dumps(dict)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("request.values--------")
        request_data = ImmutableMultiDict(request.values).get('request')
        file = request.files['file-zh[]']
        print(file)
        if file and allowed_file(file.filename):
            file.save('D:/Medical-image-segment-master/temp/1.jpg')
            # status = file.save() if status:
            result = ai.predict('D:/Medical-image-segment-master/temp/1.jpg')
            if result:

                dict = {"result": result}
                return json.dumps(dict)

    return app.send_static_file('index.html')

'''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(request.values)
        #request_data = ImmutableMultiDict(request.values).get('request')
        #dict = {"result": "1"}
        file = request.files['file-zh[]']
        print(file)
        if file and allowed_file(file.filename):
            file.save('temp/1.jpg')
            ai.predict('temp/1.jpg')
            return app.send_static_file('index.html')
'''

if __name__ == '__main__':
    app.run()
