"""A bookshop-inspired Gradio interface for semantic recommendations."""

from __future__ import annotations

import logging

import gradio as gr

from .config import TONE_TO_EMOTION
from .formatting import recommendations_to_gallery
from .recommender import BookRecommender

logger = logging.getLogger(__name__)
TONES = ["All", *TONE_TO_EMOTION.keys()]

DASHBOARD_CSS = """
.gradio-container {
    max-width: 1180px !important; margin: auto !important;
    padding: 28px 36px 24px !important; background: #f7f5ef !important;
    --body-background-fill: #f7f5ef; --body-background-fill-dark: #f7f5ef;
    --body-text-color: #263c32; --body-text-color-subdued: #59665e;
    --block-background-fill: #fffdf9; --block-background-fill-dark: #fffdf9;
    --input-background-fill: #fffdf9; --input-background-fill-dark: #fffdf9;
    --input-text-color: #263c32; --input-text-color-dark: #263c32;
    --border-color-primary: #dcded3; --border-color-primary-dark: #dcded3;
    --block-label-text-color: #435348; --block-label-text-color-dark: #435348;
    --block-title-text-color: #263c32; --block-title-text-color-dark: #263c32;
    --background-fill-primary: #fffdf9; --background-fill-secondary: #f0eee6;
    --background-fill-primary-dark: #fffdf9; --background-fill-secondary-dark: #f0eee6;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color-scheme: light;
}
body { background: #f7f5ef !important; }
#masthead { border-bottom: 1px solid #dcded3; padding-bottom: 20px; }
.brand-bar { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.brand { display: flex; align-items: center; gap: 11px; color: #203f32; }
.brand-mark { background: #244c3b; color: #faf6e9; width: 36px; height: 40px;
    display: grid; place-items: center; border-radius: 3px 10px 10px 3px; font: 24px Georgia; }
.brand-name { font: 700 25px Georgia, serif; letter-spacing: -1px; }
.brand-note { font-size: 11px; text-transform: uppercase; letter-spacing: 1.8px; color: #647164; }
.hero { display: grid; grid-template-columns: 1.5fr 1fr; align-items: center;
    gap: 40px; padding: 48px 0 38px; }
.eyebrow { font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    font-weight: 650; color: #6b765e; margin-bottom: 17px; }
.hero h1 { font: 400 clamp(38px, 5vw, 64px)/1.07 Georgia, serif !important;
    letter-spacing: -2px; color: #263e32; margin: 0 0 20px !important; }
.hero h1 em { color: #8b5539; font-weight: 400; }
.hero p { font-size: 15px; line-height: 1.8; max-width: 430px; color: #5f695f; margin: 0; }
.book-scene { height: 235px; position: relative; display: flex; align-items: flex-end;
    justify-content: center; padding-bottom: 25px; isolation: isolate; }
.book-scene:before { content: ''; position: absolute; z-index: -1; width: 255px;
    height: 225px; border-radius: 50% 50% 8px 8px; background: #e9e8db; top: 0; }
.book-scene:after { content: ''; position: absolute; bottom: 16px; width: 290px;
    height: 8px; background: #d1c7ad; border-radius: 4px; box-shadow: 0 8px 12px #263e3210; }
.book { width: 76px; height: 165px; padding: 17px 8px; color: #faf4e7; text-align: center;
    border-radius: 2px 5px 5px 2px; box-shadow: inset 5px 0 #ffffff19, 6px 5px 12px #263e3220;
    display: flex; flex-direction: column; justify-content: space-between; font: 14px Georgia; }
.book small { font: 8px sans-serif; text-transform: uppercase; letter-spacing: 1.5px; }
.book-one { background: #a65e42; transform: rotate(-12deg); margin-right: 7px; height: 150px; }
.book-two { background: #315543; transform: rotate(-3deg); z-index: 1; }
.book-three { background: #d2b675; color: #394b36; transform: rotate(10deg);
    margin-left: 7px; height: 155px; }
#search-panel { background: #fffdf9; border: 1px solid #dcded3; border-radius: 18px;
    padding: 26px; box-shadow: 0 12px 36px #324b3507; gap: 17px; }
.panel-heading { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.panel-heading h2 { font: 24px Georgia !important; color: #263e32; margin: 0 !important; }
.panel-heading span { font-size: 11px; color: #68725f; }
#book-query { border: 1px solid #dcded3; border-radius: 10px; }
#book-query textarea { font-size: 16px !important; line-height: 1.6; padding: 16px !important; }
#search-button { background: #274e3c !important; color: #fffdf7 !important;
    border: 1px solid #274e3c !important; border-radius: 9px; min-height: 47px;
    font-weight: 600; box-shadow: none; transition: background .15s, transform .15s; }
#search-button:hover { background: #193b2b !important; transform: translateY(-1px); }
#suggestions { gap: 8px; }
#suggestions button { background: #f0eee5 !important; color: #485b49 !important;
    border: 1px solid #e5e2d7 !important; border-radius: 24px;
    font-size: 12px; min-height: 35px; box-shadow: none; }
#suggestions button:hover { background: #e5e8d9 !important; border-color: #abb99f !important; }
#suggestion-label { font-size: 11px; color: #68725f; margin-bottom: -10px; }
#results-title { margin-top: 20px; }
#results-title h2 { font: 29px Georgia !important; color: #263e32; }
#result-status { color: #626d60; font-size: 13px; margin-top: -10px; }
.empty-shelf { padding: 45px 20px; text-align: center; border: 1px dashed #ced4c3;
    border-radius: 14px; background: #f0f1e7; color: #64725e; }
.empty-shelf .shelf-icon { font: 35px Georgia; letter-spacing: 5px; color: #78906d; }
.empty-shelf h3 { font: 22px Georgia !important; margin: 12px 0 8px !important; color: #435b40; }
.empty-shelf p { font-size: 13px; margin: 0; line-height: 1.7; }
#book-gallery { border: none; background: transparent; }
#book-gallery .grid-wrap { background: transparent; border: none; }
#book-gallery .thumbnail-item { background: #eeeee3; border-radius: 12px;
    border: 1px solid #dcdfd2; transition: transform .18s, box-shadow .18s; }
#book-gallery .thumbnail-item:hover { transform: translateY(-3px);
    box-shadow: 0 8px 20px #263e3215; }
#book-gallery .caption { background: #fffdf9; color: #314332; text-align: left;
    font-size: 12px; white-space: pre-line; line-height: 1.5; padding: 10px; }
.page-footer { display: flex; justify-content: space-between; gap: 12px;
    border-top: 1px solid #dcded3;
    padding: 22px 0 4px; margin-top: 24px; font-size: 11px; color: #6a7465; }
button:focus-visible, textarea:focus-visible { outline: 3px solid #aa743e !important;
    outline-offset: 3px; }
@media (max-width: 640px) {
    .gradio-container { padding: 18px 16px !important; }
    .brand-note { font-size: 9px; letter-spacing: 1px; }
    .hero { grid-template-columns: 1fr; padding: 32px 0 26px; gap: 24px; }
    .hero h1 { font-size: 43px !important; }
    .book-scene { display: none; }
    #search-panel { padding: 18px; }
    .panel-heading { align-items: flex-start; flex-direction: column; gap: 5px; }
    #book-gallery .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    .page-footer { flex-direction: column; gap: 6px; }
}
@media (prefers-reduced-motion: reduce) {
    #search-button, #suggestions button, #book-gallery .thumbnail-item { transition: none; }
}
"""

HERO = """
<section class="hero">
  <div>
    <div class="eyebrow">For your next chapter</div>
    <h1>A good book starts<br>with a <em>feeling.</em></h1>
    <p>A place for curious minds and unexpected discoveries.
    Describe the story you're craving. Find a book that feels like you.</p>
  </div>
  <div class="book-scene" aria-hidden="true">
    <div class="book book-one"><small>Find</small>
      <span>A little<br>wonder</span><span>✦</span></div>
    <div class="book book-two"><small>Your</small><span>Somewhere<br>new</span><span>✧</span></div>
    <div class="book book-three"><small>Next</small>
      <span>One more<br>chapter</span><span>✦</span></div>
  </div>
</section>
"""
EMPTY_SHELF = """
<div class="empty-shelf">
  <div class="shelf-icon" aria-hidden="true">▥ ▤ ▥</div>
  <h3>Your next chapter is waiting</h3>
  <p>Start with a feeling, a setting, or an idea.<br>
    We'll help you find your way to a good book.</p>
</div>
"""


def dashboard_theme() -> gr.themes.Soft:
    """Use local system fonts so the design does not depend on a font service."""
    return gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="stone",
        font=["system-ui", "sans-serif"],
        font_mono=["monospace"],
    )


def build_dashboard(recommender: BookRecommender) -> gr.Blocks:
    """Build the dashboard; launch with DASHBOARD_CSS and dashboard_theme()."""

    def recommend_books(query: str, category: str, tone: str) -> tuple:
        if not query or not query.strip():
            return (
                gr.update(value=[], visible=False),
                "Describe a book to get started.",
                gr.update(visible=True),
            )
        try:
            recommendations = recommender.recommend(query=query, category=category, tone=tone)
            gallery = recommendations_to_gallery(recommendations)
        except Exception:  # noqa: BLE001
            logger.exception("Recommendation failed")
            return (
                gr.update(value=[], visible=False),
                "We couldn't search the shelves just now. Please try again in a moment.",
                gr.update(visible=True),
            )
        if not gallery:
            return (
                gr.update(value=[], visible=False),
                "No matches this time. Try a broader description or choose All for the filters.",
                gr.update(visible=True),
            )
        count = len(gallery)
        return (
            gr.update(value=gallery, visible=True),
            f"{count} {'book' if count == 1 else 'books'} to explore · "
            "Select a cover for a closer look.",
            gr.update(visible=False),
        )

    with gr.Blocks(title="Margin · Find your next chapter") as dashboard:
        gr.HTML(
            '<div class="brand-bar"><div class="brand">'
            '<span class="brand-mark" aria-hidden="true">m</span>'
            '<span class="brand-name">margin</span></div>'
            '<span class="brand-note">A little discovery. A lot of possibility.</span></div>',
            elem_id="masthead",
        )
        gr.HTML(HERO)
        with gr.Column(elem_id="search-panel"):
            gr.HTML(
                '<div class="panel-heading"><h2>What are you in the mood for?</h2>'
                "<span>Follow your curiosity.</span></div>"
            )
            user_query = gr.Textbox(
                label="Describe your next read",
                show_label=False,
                placeholder="A cozy mystery in a small seaside town, with a little humor…",
                lines=2,
                max_lines=4,
                elem_id="book-query",
            )
            with gr.Row(equal_height=True):
                category = gr.Dropdown(
                    choices=recommender.categories(),
                    value="All",
                    label="Choose a category",
                    scale=2,
                    min_width=180,
                )
                tone = gr.Dropdown(
                    choices=TONES,
                    value="All",
                    label="Set the mood",
                    scale=2,
                    min_width=180,
                )
                with gr.Column(scale=2, min_width=180):
                    submit = gr.Button(
                        "Find my next read  →", variant="primary", elem_id="search-button"
                    )
            gr.Markdown("Need a spark? Try one of these", elem_id="suggestion-label")
            with gr.Row(elem_id="suggestions"):
                suggestions = [
                    (
                        "A little hope",
                        "A hopeful story about second chances and unlikely friendships",
                    ),
                    ("One more page", "An atmospheric mystery with secrets and unexpected twists"),
                    (
                        "Stay curious",
                        "An accessible nonfiction book about science and the universe",
                    ),
                ]
                buttons = [gr.Button(label, size="sm", min_width=120) for label, _ in suggestions]
        gr.Markdown("## Your reading shelf", elem_id="results-title")
        status = gr.Markdown("Good stories are worth finding.", elem_id="result-status")
        empty = gr.HTML(EMPTY_SHELF)
        gallery = gr.Gallery(
            label="Recommended books",
            show_label=False,
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain",
            elem_id="book-gallery",
            visible=False,
            interactive=False,
            buttons=[],
        )
        inputs, outputs = [user_query, category, tone], [gallery, status, empty]
        for event in (submit.click, user_query.submit):
            event(fn=recommend_books, inputs=inputs, outputs=outputs, show_progress="minimal")
        for button, (_, prompt) in zip(buttons, suggestions, strict=True):
            button.click(
                fn=lambda text=prompt: text,
                inputs=[],
                outputs=user_query,
            ).then(fn=recommend_books, inputs=inputs, outputs=outputs, show_progress="minimal")
        gr.HTML(
            '<footer class="page-footer"><span>margin · A space for your next favorite book.</span>'
            "<span>Matched by meaning. Refined by mood.</span></footer>"
        )
    return dashboard
