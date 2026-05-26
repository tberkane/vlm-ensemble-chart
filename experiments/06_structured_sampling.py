import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def encode_image_base64(image_path: Path) -> str:
    """Base64-encode image."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_type(image_path: Path) -> str:
    """Get the image type."""
    image_type = image_path.suffix.lstrip(".").lower()
    if image_type == "jpg":
        image_type = "jpeg"
    return image_type


structure = """Year\tEswatini\tKyrgyz Republic\tPhilippines
1960\t_\t_\t_
1961\t_\t_\t_
1962\t_\t_\t_
1963\t_\t_\t_
1964\t_\t_\t_
1965\t_\t_\t_
1966\t_\t_\t_
1967\t_\t_\t_
1968\t_\t_\t_
1969\t_\t_\t_
1970\t_\t_\t_
1971\t_\t_\t_
1972\t_\t_\t_
1973\t_\t_\t_
1974\t_\t_\t_
1975\t_\t_\t_
1976\t_\t_\t_
1977\t_\t_\t_
1978\t_\t_\t_
1979\t_\t_\t_
1980\t_\t_\t_
1981\t_\t_\t_
1982\t_\t_\t_
1983\t_\t_\t_
1984\t_\t_\t_
1985\t_\t_\t_
1986\t_\t_\t_
1987\t_\t_\t_
1988\t_\t_\t_
1989\t_\t_\t_
1990\t_\t_\t_
1991\t_\t_\t_
1992\t_\t_\t_
1993\t_\t_\t_
1994\t_\t_\t_
1995\t_\t_\t_
1996\t_\t_\t_
1997\t_\t_\t_
1998\t_\t_\t_
1999\t_\t_\t_
2000\t_\t_\t_
2001\t_\t_\t_
2002\t_\t_\t_
2003\t_\t_\t_
2004\t_\t_\t_
2005\t_\t_\t_
2006\t_\t_\t_
2007\t_\t_\t_
2008\t_\t_\t_
2009\t_\t_\t_
2010\t_\t_\t_
2011\t_\t_\t_
2012\t_\t_\t_
2013\t_\t_\t_
2014\t_\t_\t_
2015\t_\t_\t_
2016\t_\t_\t_
2017\t_\t_\t_
2018\t_\t_\t_
2019\t_\t_\t_
2020\t_\t_\t_
"""

CHART_EXTRACTION_PROMPT = f"""Here is an image of a chart.
Fill in the missing values (marked as "_") in the TSV skeleton below using the data from the chart.

Rules:
- Do NOT add, remove, reorder, or rename any rows or columns.
- Keep the headers EXACTLY as given: {structure.split("\n")[0].strip()}
- Replace each "_" with the correct numeric value from the chart.
- If a value is not present / cannot be determined from the chart, replace "_" with "nan".
- Use tab (\\t) as the separator.

TSV skeleton to complete (copy exactly; only replace "_" / "nan" values):
{structure}

Remember: The sole output should be the completed TSV table surrounded by ```tsv ```. Nothing else.
"""


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
image_path = Path("data/WB-ChartExtract/png/7.png")
base64_image = encode_image_base64(image_path)
image_type = get_image_type(image_path)
data_url = f"data:image/{image_type};base64,{base64_image}"
# model = "meta-llama/llama-4-maverick-17b-128e-instruct"
# response = client.responses.create(
#     model=model,
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": CHART_EXTRACTION_PROMPT},
#                 {
#                     "type": "input_image",
#                     "image_url": data_url,
#                     "detail": "high",
#                 },
#             ],
#         }
#     ],
#     temperature=1.0,
#     reasoning=None,
# )

# content = response.output_text
# if "```tsv" in content:
#     csv_data = content.split("```tsv")[1].split("```")[0].strip()
# else:
#     csv_data = content.split("```")[1].strip()

# print(csv_data)

CHART_STRUCTURE_PROMPT = """Here is an image of a chart.

Your task is to recover ONLY the table *structure* (schema + row keys), not the numeric values.

Return a TSV skeleton where:
- The first column is the x-axis field (e.g., Year).
- Each additional column is one series (use the legend/labels exactly as written).
- Include one row for every x-axis tick/category shown.
- Do NOT leave any gaps in x-axis values: if the x-axis is a numeric sequence (e.g., years), output every consecutive value from the minimum to the maximum shown (even if some ticks are not labeled).
- Put "_" for every data cell (all non-header, non-x-axis cells).
- Preserve the exact spelling, capitalization, punctuation, and ordering of headers as shown in the chart.
- Use tab characters (\\t) as separators (write real tabs, not spaces).

Output format rules:
- Output ONLY the TSV skeleton, surrounded by ```tsv
``` and nothing else.
"""


model = "gpt-5.2"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
image_path = Path("data/ChartQA/png/two_col_104053.png")
base64_image = encode_image_base64(image_path)
image_type = get_image_type(image_path)
data_url = f"data:image/{image_type};base64,{base64_image}"
response = client.responses.create(
    model=model,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": CHART_STRUCTURE_PROMPT},
                {
                    "type": "input_image",
                    "image_url": data_url,
                    "detail": "high",
                },
            ],
        }
    ],
    # reasoning={"effort": "none"},
)

content = response.output_text
if "```tsv" in content:
    csv_data = content.split("```tsv")[1].split("```")[0].strip()
else:
    csv_data = content.split("```")[1].strip()

print(csv_data)
