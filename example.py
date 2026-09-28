"""本文件提供最简示例，演示如何调用发票处理流程。"""
from pathlib import Path

from node.merge_node import process_invoice
from node.ocr_node import load_config


def main() -> None:
    """执行一次示例处理。"""
    config = load_config(Path("config.json"))
    state = process_invoice(
        input=Path("input/invoice.PDF"),
        config=config,
        truth=None,
        output_dir=Path("output/pending/example"),
        intervar=None,
        method="base",
    )

    print(f"处理完成，文档编号: {state['doc_id']}")
    print(f"错误数量: {len(state['errors'])}")
    print(f"OCR 来源: {state['metadata'].get('ocr_source', 'unknown')}")
    print(f"最终结果来源: {state['metadata'].get('final_invoice_source', 'merge')}")
    if state.get("final_invoice") or state.get("merged_invoice"):
        print("已成功提取发票结果")
        print(state.get("final_invoice") or state.get("merged_invoice"))


if __name__ == "__main__":
    main()
