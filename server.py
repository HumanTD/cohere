from coho import reader_bot,auth,get_summary,get_pdf
from flask import Flask, jsonify,request

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def home():
    if request.method == "POST": 
        # try:
        data = request.get_json()
        link = data['resume']
        company=data['companyName']
        poc=data['poc']
        role=data['role']
        get_pdf(link)
        cohere_api_key, _, _ = auth()
        pages = reader_bot('sample.pdf')
        message= get_summary(pages,cohere_api_key,company,poc,role, )

        result={
            'message':message
        }
    
        return jsonify(result)
        # except:
        #     cohere_api_key, _, _ = auth()
        #     pages = reader_bot('sample_resume.pdf')
        #     message= get_summary(pages, cohere_api_key)

        #     result={
        #         'message':message
        #     }
        
        #     return jsonify(result)
    else:
        return "GET Request not allowed."


app.run(debug=True)