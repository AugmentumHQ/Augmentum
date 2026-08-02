"""200-prompt test suite for tool filter accuracy."""
import sys
sys.path.insert(0, ".")

from collections import Counter


class FakeTool:
    def __init__(self, name, category="search"):
        self.name = name
        self.category = category
    def health_check(self):
        return True


ALL_TOOLS = [
    FakeTool("web", "search"),
    FakeTool("wikipedia", "search"),
    FakeTool("image_search", "search"),
    FakeTool("youtube_transcript", "fetch"),
    FakeTool("python_exec", "execute"),
    FakeTool("build_application", "artifact"),
    FakeTool("create_ebook", "artifact"),
    FakeTool("image_generation", "image"),
    FakeTool("file_ops", "file"),
    FakeTool("export_markdown", "artifact"),
    FakeTool("export_csv", "artifact"),
    FakeTool("export_code", "artifact"),
]

from augmentum.tools.filter import filter_tools_for_query

PROMPTS = [
    # --- Weather & Environment (10) ---
    "whats the weather today",
    "weather in tokyo",
    "is it going to rain tomorrow",
    "temperature in london right now",
    "whats the forecast for this weekend",
    "humidity levels in miami",
    "air quality index in beijing",
    "will it snow in denver",
    "sunrise time today",
    "UV index for outdoor activities",

    # --- General Knowledge Questions (20) ---
    "what is photosynthesis",
    "how does gravity work",
    "why is the sky blue",
    "what causes earthquakes",
    "who invented the telephone",
    "where is the Great Wall of China",
    "when was the Roman Empire founded",
    "how tall is the Eiffel Tower",
    "what is the speed of light",
    "who discovered penicillin",
    "what is the largest ocean",
    "how far is the moon from earth",
    "what does DNA stand for",
    "why do leaves change color in fall",
    "how many bones are in the human body",
    "who painted the Mona Lisa",
    "what is the capital of Australia",
    "how old is the universe",
    "where is Timbuktu",
    "what is machine learning",

    # --- Current Events & News (10) ---
    "latest news about AI",
    "what happened in the stock market today",
    "recent developments in quantum computing",
    "who won the election",
    "current gas prices",
    "breaking news today",
    "trending topics on social media",
    "latest SpaceX launch",
    "current COVID situation",
    "recent supreme court rulings",

    # --- Math & Calculations (10) ---
    "what is 247 times 389",
    "calculate the area of a circle with radius 5",
    "solve x^2 - 4 = 0",
    "convert 100 fahrenheit to celsius",
    "how many miles in a kilometer",
    "calculate compound interest on 10000 at 5 percent for 10 years",
    "what is the square root of 144",
    "factorial of 12",
    "convert 50 pounds to kilograms",
    "what is 15 percent of 230",

    # --- Code & Programming (10) ---
    "write a python function to sort a list",
    "debug this javascript code",
    "how to implement a binary search",
    "regex to match email addresses",
    "write a script to rename files",
    "implement quicksort algorithm",
    "how to parse JSON in python",
    "run this code and tell me the output",
    "execute this python script for me",
    "compile this C program",

    # --- Image Generation (10) ---
    "draw a picture of a sunset",
    "generate an image of a cat wearing a hat",
    "create an illustration of a fantasy castle",
    "make me a picture of a robot",
    "visualize what mars looks like",
    "render a 3D scene of mountains",
    "sketch a portrait of Einstein",
    "generate a logo for my company",
    "draw a diagram of the solar system",
    "create an image of a futuristic city",

    # --- App Building (10) ---
    "build me a todo app",
    "create a calculator web app",
    "make a simple game",
    "build a dashboard for sales data",
    "create a web page for my portfolio",
    "build me a timer application",
    "make a weather app",
    "create a quiz game",
    "build a form for customer feedback",
    "make me a simple chat interface",

    # --- Ebook & Stories (10) ---
    "write a childrens storybook about a dragon",
    "create an illustrated ebook about space",
    "write a short story and make it an epub",
    "create a picture book for kids",
    "write a fairy tale with illustrations",
    "make an ebook about cooking basics",
    "illustrated book about ocean animals",
    "write a bedtime story",
    "create a storybook about friendship",
    "make an epub novel",

    # --- YouTube (10) ---
    "summarize this youtube video https://youtube.com/watch?v=abc123",
    "get the transcript of this video https://youtu.be/xyz789",
    "what does this youtube video say",
    "transcript of the latest TED talk",
    "captions for this video",
    "subtitle extraction from youtube",
    "summarize the youtube video about AI",
    "get video captions https://youtube.com/watch?v=test",
    "what is this youtuber talking about https://youtube.com/watch?v=demo",
    "youtube transcript please",

    # --- File Operations (5) ---
    "read the contents of config.json",
    "write this text to a file called notes.txt",
    "list all files in the project",
    "save this to a file",
    "open the readme file",

    # --- Export (5) ---
    "export this conversation as markdown",
    "save this data as a CSV file",
    "download this as a .py file",
    "export the results to a file",
    "save as markdown please",

    # --- Image Search (5) ---
    "find an image of the Eiffel Tower",
    "search for a photo of a golden retriever",
    "find a picture of the northern lights",
    "image of the periodic table",
    "search for a diagram of a cell",

    # --- Wikipedia (5) ---
    "look up Albert Einstein on wikipedia",
    "wikipedia article about quantum physics",
    "wiki page for the French Revolution",
    "encyclopedia entry on democracy",
    "biography of Marie Curie",

    # --- Conversational / No Tools Needed (20) ---
    "hello how are you",
    "tell me a joke",
    "what do you think about life",
    "can you help me brainstorm ideas",
    "summarize this text for me",
    "translate this to Spanish",
    "rewrite this paragraph to be more concise",
    "proofread my essay",
    "give me advice on time management",
    "help me write an email",
    "compare these two options for me",
    "what are the pros and cons",
    "explain this concept simply",
    "rate my resume",
    "suggest a good book",
    "help me with my homework",
    "write a poem about nature",
    "create a meal plan for the week",
    "role play as a pirate",
    "lets play 20 questions",

    # --- Ambiguous / Edge Cases (20) ---
    "make me something cool",
    "I need help",
    "do something with python",
    "can you look something up for me",
    "analyze this data",
    "process this information",
    "what can you do",
    "show me something interesting",
    "give me a report on climate change",
    "research quantum computing and make a presentation",
    "find information and create a document",
    "search for recipes and export as pdf",
    "build a website that shows the weather",
    "create a chart showing population growth",
    "generate images for my book about cats",
    "write code to solve this math problem",
    "get the youtube transcript and summarize it",
    "look up the stock price and calculate returns",
    "what is 2+2",
    "hi",

    # --- Multi-intent (10) ---
    "search for AI news then write a report",
    "find an image and create a presentation",
    "calculate my taxes and export to csv",
    "research this topic and build an app about it",
    "get weather data and make a chart",
    "write a story illustrate it and make an ebook",
    "look up recipes and create a cookbook epub",
    "search for data analyze it with python export csv",
    "find youtube video about cooking and summarize",
    "compare products and create a spreadsheet",
]


def main():
    results = []
    for prompt in PROMPTS:
        filtered = filter_tools_for_query(prompt, ALL_TOOLS, min_tools=2, max_tools=8)
        tool_names = sorted([t.name for t in filtered])
        results.append({
            "prompt": prompt,
            "tools": tool_names,
            "count": len(tool_names),
        })

    print(f"Tested {len(results)} prompts against {len(ALL_TOOLS)} tools\n")

    # Stats
    has_build = sum(1 for r in results if "build_application" in r["tools"])
    has_ebook = sum(1 for r in results if "create_ebook" in r["tools"])
    has_web = sum(1 for r in results if "web" in r["tools"])
    has_python = sum(1 for r in results if "python_exec" in r["tools"])
    has_image_gen = sum(1 for r in results if "image_generation" in r["tools"])
    all_tools = sum(1 for r in results if r["count"] == len(ALL_TOOLS))
    avg_count = sum(r["count"] for r in results) / len(results)

    print("=== SUMMARY ===")
    print(f"Average tools per query: {avg_count:.1f} (of {len(ALL_TOOLS)} total)")
    print(f"Queries returning ALL tools: {all_tools}")
    print(f"Queries with build_application: {has_build}")
    print(f"Queries with create_ebook: {has_ebook}")
    print(f"Queries with web: {has_web}")
    print(f"Queries with python_exec: {has_python}")
    print(f"Queries with image_generation: {has_image_gen}")
    print()

    # Tool frequency
    tool_freq = Counter()
    for r in results:
        for t in r["tools"]:
            tool_freq[t] += 1
    print("=== TOOL FREQUENCY ===")
    for name, count in tool_freq.most_common():
        bar = "#" * (count // 3)
        print(f"  {name:25s} {count:3d}/200  {bar}")
    print()

    # Queries where build_application appears
    print("=== QUERIES WITH build_application ===")
    for r in results:
        if "build_application" in r["tools"]:
            print(f"  [{r['count']} tools] {r['prompt']!r}")
            print(f"           -> {r['tools']}")
    print()

    # Queries returning ALL tools
    print("=== QUERIES RETURNING ALL TOOLS (filter miss) ===")
    for r in results:
        if r["count"] == len(ALL_TOOLS):
            print(f"  {r['prompt']!r}")
    if not any(r["count"] == len(ALL_TOOLS) for r in results):
        print("  (none)")
    print()

    # Conversational queries
    print("=== CONVERSATIONAL (should be minimal) ===")
    conv_start = PROMPTS.index("hello how are you")
    for r in results:
        if r["prompt"] in PROMPTS[conv_start:conv_start + 20]:
            print(f"  [{r['count']} tools] {r['prompt']!r} -> {r['tools']}")
    print()

    # Ambiguous queries
    print("=== AMBIGUOUS / EDGE CASES ===")
    amb_start = PROMPTS.index("make me something cool")
    for r in results:
        if r["prompt"] in PROMPTS[amb_start:amb_start + 20]:
            print(f"  [{r['count']} tools] {r['prompt']!r} -> {r['tools']}")


if __name__ == "__main__":
    main()
