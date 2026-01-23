from flask import Flask, Blueprint

# 1. Flask 앱 팩토리 함수 정의
def create_app():
    app = Flask(__name__)
    
    # 기본 설정 (추후 필요시 확장)
    app.config['SECRET_KEY'] = 'dev-key-1234'
    
    # 2. Blueprint 등록 (추후 routes 폴더와 연결할 부분)
    # 현재는 예시로 빈 Blueprint를 내부에서 정의하지만, 
    # 나중에 'from routes.main import main_bp' 형태로 불러와서 등록하면 됩니다.
    
    # [임시 Blueprint 정의] - 나중에 삭제하고 외부 파일에서 가져오세요
    main_bp = Blueprint('main', __name__)
    
    @main_bp.route('/')
    def home():
        return "<h3>HBM Project Flask Server (Blueprint Setup)</h3>"
    
    # Blueprint를 앱에 등록
    app.register_blueprint(main_bp)
    
    return app

# 3. 앱 인스턴스 생성
app = create_app()

if __name__ == "__main__":
    # 도커/로컬 환경 모두 대응 가능한 실행 설정
    print(" >>> Starting Flask Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
