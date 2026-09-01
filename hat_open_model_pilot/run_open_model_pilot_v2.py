import run_open_model_pilot as p

def ids_compat(tok, msgs):
    x = tok.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(x, "input_ids"):
        return x.input_ids
    if isinstance(x, dict) and "input_ids" in x:
        return x["input_ids"]
    return x

p.ids = ids_compat

if __name__ == "__main__":
    p.main()
