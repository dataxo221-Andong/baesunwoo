from flask import Flask
from routes import register_routes

def create_app():
    app = Flask(__name__)
    
    # 기본 설정
    app.config['SECRET_KEY'] = 'dev-key-1234'
    
    # 라우트 등록 (routes/__init__.py 내의 함수 호출)
    register_routes(app)
    
    # [추가] 루트('/') 접속 시 '/main/'으로 리다이렉트
    from flask import redirect
    @app.route('/')
    def root():
        return redirect('/main/')
    
    return app

app = create_app()

if __name__ == "__main__":
    print(" >>> Starting Flask Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
