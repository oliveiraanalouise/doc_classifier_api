## Versão Python: 3.10.0

1. Crie e ative um ambiente virtual

```bash
python -m venv .venv

source .venv/bin/activate
```

2. Instale as dependências

```bash
pip install -r requirements.txt
```

3. Execute o script 
```bash
python api.py
```

4. Acesse o dashboard no link http://localhost:5000/api/docs

5. Envie uma requisição
Para o correto funcionamento, envie uma requisição tipo multipart, onde exista um campo "file" com o arquivo anexado a esse campo.
