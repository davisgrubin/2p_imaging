#!/usr/bin/env python3

import os
import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_logo_html():
    ellie = encode_image_to_base64(os.path.join('webpage', 'ellie.gif'))
    bme_logo = encode_image_to_base64(os.path.join('webpage', 'bme_logo.svg'))
    logo_html = f"""
        <a href="https://www.bilibili.com/video/BV1fY4y1F7GL" target="_blank">
            <img src="data:image/gif;base64,{ellie}" title="
                Remember:
                Mice will randomly die.
                Experiments will go shit.
                Papers will get rejected.
                Farzaneh will kill you.
                BUT YICONG WILL LOVE YOU FOREVER!"
                style="position:absolute; width:80px; height:auto;">
        </a>
        <div style="text-align:center;">
            <a href="https://bme.gatech.edu/bme/" target="_blank">
                <img src="data:image/svg+xml;base64,{bme_logo}" title="I AM CUTE! GIVE ME A HUG!"
                    style="max-width:50%; height:auto;">
            </a>
        </div>
        <hr style="border:0.1px solid #ccc; margin: 1px 0;" />
        """
    return logo_html