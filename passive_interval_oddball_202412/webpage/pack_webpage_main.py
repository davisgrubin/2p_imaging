import os
import base64
import shutil

from webpage.html_session_list import get_session_list_html
from webpage.html_dropdown_menu import get_dropdown_menu_html
from webpage.html_logo import get_logo_html
from webpage.css_styles import get_css_styles
from webpage.java_scripts import get_java_scripts

# main process to combine results.
def run(session_config_list, list_fn, list_page_name, list_target_sess):
    
    # helper function to embed figures into an html block.
    def generate_div(filename):
        scale_pixel = 196
        # read svg content.
        with open(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename[0]+'.svg'),
                  'r', encoding="utf-8") as svg_file:
            svg_str = svg_file.read()
        svg_file.close()
        with open(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename[0]+'.svg'),
                  'rb') as svg_file:
            svg_bytes = svg_file.read()
            svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')
        svg_file.close()
        # read discription in txt file.
        try:
            with open(os.path.join('results', 'notes', filename[0]+'.txt'),
                      'r', encoding='utf-8') as text_file:
                discription = text_file.read()
            text_file.close()
        except:
            discription = ''
        # create a figure block for a svg file.
        html_block = f"""
        <div class="figure-block">
            <div class="left-panel">
                <div class="description">
                    {discription}
                </div>
            </div>
                <div class="right-panel">
                    <div class="figure-title">
                        <a href="data:application/svg;base64,{svg_b64}"
                            download="{os.path.basename(filename[0]+'.svg')}"
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
        os.remove(os.path.join('results', 'temp_'+session_config_list['subject_name'], filename[0]+'.svg'))
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
    def get_full_html(session_config_list, table_html, pages_html):
        # generate other codes.
        dropdown_menu_html = get_dropdown_menu_html(list_page_name)
        logo_html = get_logo_html()
        css_styles = get_css_styles()
        java_scripts = get_java_scripts(len(list_page_name))
        # final html output code.
        html_output = f"""
        <html>
            <head>
                <meta charset="utf-8"/>
                <title>Passive session report for {session_config_list['subject_name']}</title>
                {css_styles}
                {java_scripts}
            </head>
            <body>
                {logo_html}
                <h1 style="text-align:left;">Passive session report for {session_config_list['subject_name']}</h1>
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
    for pi in range(len(list_page_name)):
        pages_html += get_page_html(list_fn[pi], pi, list_page_name[pi])
    # generate list of session name html codes.
    session_list_html = get_session_list_html(session_config_list, list_target_sess)
    # get full html output.
    html_output = get_full_html(session_config_list, session_list_html, pages_html)
    # save result.
    output_path = os.path.join('results', session_config_list['output_filename']+'.html')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)
    # delete temp folder.
    shutil.rmtree(os.path.join('results', 'temp_'+session_config_list['subject_name']))

