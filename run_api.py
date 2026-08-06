from flask import Flask, request, Response
from superlm import WorkSpace
app = Flask(__name__)
workspace = WorkSpace('backrooms')

@app.route('/api', methods=['GET'])
def api():
    args = dict(request.args)
    prompt = args.pop('prompt', '')
    return Response(
        workspace.api(prompt, **args),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache'}
    )

if __name__ == '__main__':
    app.run()