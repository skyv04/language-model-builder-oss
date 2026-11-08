// Pre-training data: pick (or import) the plain-text corpus the model learns language from.
import { dataPage } from "../workbench/data.js";

export default dataPage({ id: "pretraining-data", title: "Pre-training data", kind: "pretrain" });
