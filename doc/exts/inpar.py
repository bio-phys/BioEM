def setup(app):
    app.add_crossref_type('inpar', 'inpar', 'single: %s')
    app.add_crossref_type('outpar', 'outpar', 'single: %s')    

    return {'version': '0.1'}   # identifies the version of our extension
