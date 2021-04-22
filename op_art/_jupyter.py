import pkgutil

def init_jupyter():
    from IPython.display import display, Javascript, HTML
    js_data = pkgutil.get_data(__name__, "web/op_art.js").decode()
    css_data = pkgutil.get_data(__name__, "web/op_art.css").decode()

    css_html_data = f"<style>\n{css_data}\n</style>"

    display(Javascript("require.config({paths: {d3: 'https://d3js.org/d3.v5.min'}});"))
    display(Javascript(data=js_data))
    display(HTML(data=css_html_data))