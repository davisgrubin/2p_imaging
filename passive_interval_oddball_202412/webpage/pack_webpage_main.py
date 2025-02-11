import os
import base64

from webpage.html_session_list import get_session_list_html
from webpage.html_dropdown_menu import get_dropdown_menu_html
from webpage.html_logo import get_logo_html
from webpage.css_styles import get_css_styles
from webpage.java_scripts import get_java_scripts

# main process to combine results.
def run(session_config, fn1, fn2, fn3, fn4):
    
    # helper function to embed figures into an html block.
    def generate_div(filename):
        scale_pixel = 256
        # read svg content.
        with open(os.path.join('results', session_config['subject_name']+'_temp', filename[0]+'.svg'),
                  'r', encoding="utf-8") as svg_file:
            svg_str = svg_file.read()
        svg_file.close()
        # read pdf content and encode in base64.
        with open(os.path.join('results', session_config['subject_name']+'_temp', filename[0]+'.pdf'),
                  'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_file.close()
        # read discription in txt file.
        try:
            with open(os.path.join('results', 'notes', filename[0]+'.txt'),
                      'r', encoding='utf-8') as text_file:
                discription = text_file.read()
            text_file.close()
        except:
            discription = ''
        # create a figure block for a pdf file.
        html_block = f"""
        <div class="figure-block">
            <div class="left-panel">
                <div class="description">
                    {discription}
                </div>
            </div>
                <div class="right-panel">
                    <div class="figure-title">
                        <a href="data:application/pdf;base64,{pdf_b64}"
                            download="{os.path.basename(filename[0]+'.pdf')}"
                            class="download-title"> {filename[3]}
                        </a>
                    </div>
                    <div class="svg-container"
                        style="width: {filename[2]*scale_pixel}px; height: {filename[1]*scale_pixel}px;">
                        {svg_str}
                    </div>
                </div>
        </div>
        """
        # clear space.
        os.remove(os.path.join('results', session_config['subject_name']+'_temp', filename[0]+'.svg'))
        os.remove(os.path.join('results', session_config['subject_name']+'_temp', filename[0]+'.pdf'))
        return html_block
    
    # create html code for a page given list of figure filenames.
    def get_page_html(filenames, page_i, page_name):
        display_style = "" if page_i == 0 else "display:none;"
        if len(filenames) > 0:
            # generate figure block html.
            blocks_html = ""
            for filename in filenames:
                blocks_html += generate_div(filename=filename)
            # combine into page html.
            page_html = f"""
            <div id="page{page_i+1}" class="page-container" style="{display_style}">
                <h2>{page_name}</h2>
                {blocks_html}
            </div>
            """
        else:
            page_html = f"""
            <div id="page{page_i+1}" class="page-container" style="{display_style}">
                <h2>{page_name}</h2>
                <p style="color: #666; font-size: 0.9em;">
                    No available session found.
                </p>
            </div>
            """
        return page_html
    
    # finalize all html output.
    def get_full_html(session_config, table_html, pages_html):
        n_pages = 4
        # generate other codes.
        dropdown_menu_html = get_dropdown_menu_html()
        logo_html = get_logo_html()
        css_styles = get_css_styles()
        java_scripts = get_java_scripts(n_pages)
        # final html output code.
        html_output = f"""
        <html>
            <head>
                <meta charset="utf-8"/>
                <title>Passive session report for {session_config['subject_name']}</title>
                {css_styles}
                {java_scripts}
            </head>
            <body>
                {logo_html}
                <h1 style="text-align:left;">Passive session report for {session_config['subject_name']}</h1>
                {table_html}
                <hr style="border:1px solid #ccc; margin: 10px 0;"/>
                {dropdown_menu_html}
                <hr style="border:0.1px solid #ccc; margin: 1px 0;"/>
                {pages_html}
            </body>
        </html>
        """
        return html_output
    
    # generate page containers and assign blocks to each page.
    pages_html = ""
    pages_html += get_page_html(fn1, 0, 'field_of_view')
    pages_html += get_page_html(fn2, 1, 'random')
    pages_html += get_page_html(fn3, 2, 'short_long')
    pages_html += get_page_html(fn4, 3, 'fix_jitter_odd')
    # generate list of session name html codes.
    session_list_html = get_session_list_html(session_config)
    # get full html output.
    html_output = get_full_html(session_config, session_list_html, pages_html)
    # save result.
    output_path = os.path.join('results', session_config['output_filename'])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)
    # delete temp folder.
    os.rmdir(os.path.join('results', session_config['subject_name']+'_temp'))

