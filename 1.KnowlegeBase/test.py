

import os
import fitz


if __name__ == "__main__":
    
    files = ['data/2022职业大典-第四类.pdf']

    for file in files:

        doc = fitz.open(file)
        for page in doc :
            text = page.get_text()
            print(text)