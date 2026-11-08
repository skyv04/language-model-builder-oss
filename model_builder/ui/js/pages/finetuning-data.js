// Fine-tuning data: pick (or import) the conversations the model practises answering.
import { dataPage } from "../workbench/data.js";

export default dataPage({ id: "finetuning-data", title: "Fine-tuning data", kind: "finetune" });
