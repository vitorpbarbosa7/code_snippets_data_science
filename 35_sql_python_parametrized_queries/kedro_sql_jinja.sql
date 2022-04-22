from jinja2 import Template

%load_ext kedro.extras.extensions.ipython
%reload_kedro

alldata = catalog.load('alldata')

query = catalog.load('query')

print(query)

ref = '202202'
particao_publico = 'teste_v1'

params = {
    'ref': ref,
    'particao_publico': particao_publico
}

query_template = Template(query)

query_final = query_template.render(params)

print(query_final)
