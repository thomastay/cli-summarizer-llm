from dataclasses import dataclass


@dataclass
class Args:
    type: str
    display_prompt: bool
    no_generate: bool
    include_code: bool
    include_tables: bool
