// Reading-pane guides for the workbench pages. Plain HTML strings; app.js shows the one for the current page.
export const guides = {
  "project-setup": `
<p>This page decides what your model is: how text is cut into tokens, how big the network is, and which chip trains it. Everything stays editable until pre-training saves its first checkpoint. From then on the shape and tokenizer are locked.</p>
<h3>Pick a shape</h3>
<ul>
<li><strong>Pocket</strong> (a 128-token context, width 128, 4 blocks) is the one for phones and machines with little free memory. It trains fastest, so it is ideal for a first trip through every stage.</li>
<li><strong>Starter</strong>, the default for new projects, is a balanced first model. <strong>Large</strong> writes noticeably better text but costs several times as much per step.</li>
<li><strong>Custom</strong> unlocks the sliders. <strong>Attention heads</strong> only offers counts that split the width evenly.</li>
</ul>
<p>The line under the preset shows the parameter count and the cost of one step compared with a Starter step. If the blueprint shows a memory warning, choose a smaller shape or batch size, or close other apps.</p>
<h3>Choose a tokenizer</h3>
<p><strong>Fast BPE</strong> (recommended) is learned from your own pre-training text when the first run starts, with 4K, 10K or 16K entries. <strong>GPT-2 BPE</strong> needs no training, but its embedding table has 50,261 rows and can outweigh the rest of a small model.</p>
<h3>Measure this machine</h3>
<ol>
<li>Press <strong>Calibrate</strong> in the blueprint. It times a few training steps on every device the app found: the CPU, plus any GPU it can reach through CUDA, Metal (on a Mac) or Vulkan (for example an Adreno GPU with Mesa's Turnip driver).</li>
<li><strong>Training speed</strong> then lists seconds per step and a time range for a standard run on each device.</li>
<li>Leave <strong>Compute device</strong> on <strong>Auto</strong> to train on whichever device measured faster, or pick <strong>CPU</strong> or <strong>GPU</strong> yourself.</li>
</ol>
<p>A measurement belongs to one shape, vocabulary and batch size, so calibrate again after changing any of them. Until you do, Auto uses a CUDA GPU or an Apple silicon GPU if there is one, and the CPU otherwise. A device change takes effect from the next run.</p>
<p>To start over with a different shape, use <strong>Reset pre-training…</strong> in the lock banner. It deletes every checkpoint of the project. Background: <a href="#/tokenization">Tokenization</a> and <a href="#/anatomy-of-a-transformer">Anatomy of a Transformer</a>.</p>
`,

  "pretraining-data": `
<p>Pre-training teaches the model what ordinary text looks like, so this choice colours everything it writes later. Each project learns from one corpus.</p>
<h3>Choose from the catalog</h3>
<ul>
<li>Click a tile in the <strong>Dataset catalog</strong> to see its size, source and license, plus a preview of <strong>What your model will read</strong>.</li>
<li><strong>Download &amp; use</strong> fetches the dataset and makes it this project's data. If it is already on disk, the button reads <strong>Use this dataset</strong>.</li>
<li>A chosen dataset that isn't on disk yet is downloaded when pre-training starts.</li>
</ul>
<h3>Which one?</h3>
<ul>
<li><strong>TinyStories</strong> (recommended) suits small models best. Its simple words, short sentences and repeated patterns are learnable even for a Pocket model.</li>
<li><strong>Tiny Shakespeare</strong> is small and distinctive. That makes it fun, but a model can soon start memorising it.</li>
<li>Encyclopedia-style sets such as WikiText-2 or Simple English Wikipedia cover far more ground. They need a bigger model and more steps before the output reads well.</li>
</ul>
<h3>Bring your own text</h3>
<p><strong>Add your own</strong> imports a plain-text file (.txt or .md). Blank lines separate documents, and the file never leaves this computer. Only use text you have the right to use.</p>
<h3>Good to know</h3>
<ul>
<li>One document in fifty is held back as validation text. That is where the <strong>Val</strong> loss on the Pre-training page comes from.</li>
<li>The Fast BPE tokenizer is trained on this corpus when the first run starts.</li>
<li>The Pre-training page shows how many passes over the corpus your step target adds up to.</li>
<li><strong>Delete downloaded files…</strong> in the <strong>More actions</strong> menu frees the disk space. The dataset stays in the catalog.</li>
</ul>
<p>Background: <a href="#/training-data">Training data</a>.</p>
`,

  pretraining: `
<p>Here the model learns language from scratch. It reads your corpus in batches, and after each batch it nudges its weights so the next token becomes a little easier to predict.</p>
<h3>Run it</h3>
<ol>
<li>Check the pipeline: Data, Tokenizer, Model, This run. <strong>Start pre-training</strong> runs the whole chain. It downloads the corpus if needed, trains the tokenizer and turns the text into token ids, then trains.</li>
<li>Watch the curve. <strong>Loss</strong> starts near the logarithm of the vocabulary size (about 9.2 for 10K tokens), falls quickly, then flattens. <strong>Val</strong> is measured on held-out text every 100 steps.</li>
<li><strong>Pause</strong> saves a checkpoint. <strong>Resume pre-training</strong> continues from it.</li>
</ol>
<h3>Run length</h3>
<ul>
<li><strong>Step target</strong> stops when the step counter reaches the slider value (2K by default). The time estimate follows the pace of the last run, or your calibration if nothing has run yet.</li>
<li><strong>Until I stop</strong> keeps going until you press Pause. After warmup, the learning rate then stays at its peak.</li>
<li>When a run is complete, you can raise the target and resume. The decay stretches to the new target, so the learning rate rises a little again.</li>
</ul>
<h3>Recipe</h3>
<ul>
<li><strong>Batch size</strong> (16) is the number of sequences read per step. Bigger batches give smoother steps but cost more memory and time.</li>
<li><strong>Learning rate</strong> (6e-4) is the peak step size. If the loss spikes or climbs, lower it.</li>
<li><strong>Warmup</strong> (100) is how many steps the rate takes to ramp up. After that it follows a cosine curve down to a tenth of its peak at the target.</li>
</ul>
<h3>Tips</h3>
<ul>
<li>The <strong>Sample timeline</strong> keeps a short sample from every 250 steps, so you can watch the writing improve.</li>
<li>If Val stops falling while Loss keeps dropping, the model is memorising. Use more data or stop earlier.</li>
<li>With <strong>Keep training after closing the window</strong> turned on in Settings (the default), runs carry on in the background. With it off, closing the window pauses the run.</li>
</ul>
<p>Background: <a href="#/how-models-learn">How models learn</a>.</p>
`,

  sampling: `
<p>Sampling lets you prompt any saved checkpoint and watch it write, one token at a time. Because it reads checkpoints, it also works while a run is going. It is just a little slower then, because the two share the machine.</p>
<h3>Prompt and generate</h3>
<ul>
<li>Type the opening of something, or start from a suggestion such as <strong>Story</strong>, <strong>Q&amp;A</strong> or <strong>Chat-like</strong>. The model carries on exactly where your words stop.</li>
<li><strong>Generate</strong> (Ctrl+Enter) writes up to <strong>Max tokens</strong>. <strong>Continue</strong> keeps going from the end, and <strong>Retry</strong> writes it again with a new seed.</li>
<li>Once more than one stage is trained, switch between the <strong>Base</strong>, <strong>SFT</strong> and <strong>DPO</strong> checkpoints at the top of the card.</li>
</ul>
<h3>Look inside</h3>
<ul>
<li>The magnifier button shows each token as a chip, and brighter chips were likelier picks. Click a chip to see the runners-up. Choosing one of them rewrites the text from that point on.</li>
<li><strong>Next token</strong> lists the model's top candidates for what comes next. Click one to write it in, then press Continue.</li>
</ul>
<h3>Decoding</h3>
<ul>
<li><strong>Predictable</strong>, <strong>Balanced</strong> and <strong>Creative</strong> are ready-made mixes. Moving any slider switches to a custom mix.</li>
<li><strong>Temperature</strong> sharpens or flattens the odds. <strong>Top-k</strong>, <strong>Top-p</strong> and <strong>Min-p</strong> each remove unlikely tokens before the draw.</li>
<li>Lock the seed to make runs repeatable, then change one setting at a time. <strong>Compare all three presets</strong> writes one prompt with one seed three ways, side by side.</li>
</ul>
<p>Early checkpoints babble, and that is expected. Come back every few hundred steps and try the same prompt and seed again. Your latest runs are kept under <strong>Recent samples</strong>.</p>
<p>Background: <a href="#/what-is-a-language-model">What is a language model?</a></p>
`,

  "finetuning-data": `
<p>Fine-tuning data is a set of example conversations. Supervised fine-tuning trains the model to write the assistant's side of them. That is what turns something that continues text into something that takes turns.</p>
<h3>Choose a dataset</h3>
<ul>
<li><strong>Everyday Conversations</strong> (recommended) is short, friendly small talk. It suits a small model that was pre-trained on simple stories.</li>
<li>Other sets teach a particular habit: GSM8K maths word problems, CoEdIT text editing, Code Alpaca code, and SQuAD answering from a passage. No Robots, Dolly, OpenAssistant, SmolTalk and UltraChat are general assistant chats. A phone-sized model copies their format and tone long before it masters the skill.</li>
<li>The <strong>Example training row</strong> preview shows a conversation the model will learn from. Page through a few.</li>
</ul>
<h3>Bring your own</h3>
<p><strong>Add your own</strong> takes a .jsonl, .json or .csv file. Each row can hold a <code>messages</code> list of <code>role</code>/<code>content</code> turns, or a pair of fields such as <code>prompt</code>/<code>response</code>, <code>instruction</code>/<code>output</code> or <code>question</code>/<code>answer</code>. The file stays on this computer.</p>
<h3>How it is used</h3>
<ul>
<li>Every conversation is wrapped in the chat template. Its <code>&lt;|system|&gt;</code>, <code>&lt;|user|&gt;</code> and <code>&lt;|assistant|&gt;</code> turns each end with <code>&lt;|end|&gt;</code>. Rows without a system turn get the default one: “You are a friendly assistant. Give clear, short answers.”</li>
<li>Only the assistant's replies are graded, each with its closing <code>&lt;|end|&gt;</code>. Everything else is context.</li>
<li>One conversation in twenty (at most 500) is held back for the validation loss.</li>
<li>Conversations longer than the context window are cut off where the window ends. Short exchanges work best with a small window such as Pocket's 128 tokens.</li>
</ul>
<p>Background: <a href="#/how-models-learn-behavior">How models learn behavior</a>.</p>
`,

  sft: `
<p>Supervised fine-tuning copies your latest pre-training checkpoint and keeps training it on conversations. From them the model learns the shape of a chat: read the user's turn, write a reply, then close it with <code>&lt;|end|&gt;</code>. The base model itself stays untouched.</p>
<h3>Before you start</h3>
<ul>
<li><strong>Starts from</strong> shows the pre-training step it builds on, and <strong>Dataset</strong> shows the conversations it will practise. Both have to be set.</li>
<li>To fine-tune a newer base later, press <strong>Reset</strong> and start again. Resetting this stage also clears preference tuning.</li>
</ul>
<h3>Run it</h3>
<ol>
<li>Press <strong>Start fine-tuning</strong>. The loss counts only the assistant's tokens. It usually drops quickly at first and then levels off.</li>
<li>Every 100 steps the <strong>Sample timeline</strong> asks one of six built-in questions, so you can watch the replies take shape.</li>
<li><strong>Pause</strong> saves a checkpoint, and <strong>Resume fine-tuning</strong> continues from it.</li>
</ol>
<h3>Recipe</h3>
<ul>
<li><strong>Run length</strong> defaults to 400 steps. The arrows move between 50 and 5,000.</li>
<li>The default <strong>Learning rate</strong>, 1e-4, is six times gentler than pre-training's. That way the model picks up the format without overwriting what it already knows.</li>
<li><strong>Warmup</strong> is 20 steps, and <strong>Batch size</strong> is 8 conversations per step.</li>
</ul>
<h3>Reading the result</h3>
<ul>
<li>Compare <strong>Val</strong> with <strong>Loss</strong>. On a small dataset the model soon starts memorising its examples. Once Val stops improving, more steps mostly teach it to parrot them.</li>
<li>Tiny models learn the format long before they learn facts. Short, on-topic replies that end cleanly count as a success.</li>
<li>Try the result on the Chat page, switching between <strong>Base</strong> and <strong>SFT</strong> to see what changed.</li>
</ul>
<p>Background: <a href="#/how-models-learn-behavior">How models learn behavior</a>.</p>
`,

  dpo: `
<p>Direct preference optimization nudges your fine-tuned model towards the answers you prefer. You vote on pairs of its own answers. The tuning run then makes the chosen answers more likely and the rejected ones less likely, compared with a frozen copy of the fine-tuned model.</p>
<h3>Fill the ballot</h3>
<ol>
<li><strong>Sample two answers</strong> asks the fine-tuned model the same question twice, with loose enough sampling that the two answers differ. The pencil button lets you write your own question, and the shuffle button picks another built-in one.</li>
<li>Read <strong>Candidate A</strong> and <strong>Candidate B</strong>, then press <strong>This one is better</strong> on the one you would rather get. The pair is banked and a new pair is sampled.</li>
<li>The tuning run unlocks once 4 pairs are banked. More pairs give a steadier signal.</li>
</ol>
<h3>Tune</h3>
<ul>
<li><strong>Start tuning</strong> runs 60 steps by default (10 to 500), with batch size 4 and learning rate 2e-5 after a short warmup.</li>
<li><strong>Beta</strong> (0.01 to 0.5, default 0.1) is the leash. A low beta lets the model drift further from the fine-tuned version, and a higher one keeps it close.</li>
<li>The loss starts at ln 2 ≈ 0.69, which means no preference yet. It falls as the model sides with your picks.</li>
</ul>
<h3>Things to watch</h3>
<ul>
<li>With only a few pairs, the model can latch onto accidents such as answer length. Vary your questions and vote consistently.</li>
<li>Tuning runs on the CPU, a CUDA GPU or a Mac's GPU through Metal. When a Vulkan GPU is selected, it uses the CPU instead.</li>
<li><strong>Pause</strong> saves a checkpoint, and <strong>Resume tuning</strong> continues from it. The fine-tuned model is never changed.</li>
<li><strong>Clear</strong> under <strong>Banked pairs</strong> deletes your votes but keeps any tuning checkpoints.</li>
<li>Compare <strong>SFT</strong> and <strong>DPO</strong> side by side on the Chat page.</li>
</ul>
<p>Background: <a href="#/how-models-learn-behavior">How models learn behavior</a>.</p>
`,

  chat: `
<p>Chat puts your model in a conversation. Each message is wrapped in the same chat template the model was fine-tuned on. The reply is written one token at a time, until the model writes <code>&lt;|end|&gt;</code> or runs out of room.</p>
<h3>Pick a checkpoint</h3>
<ul>
<li><strong>Base</strong> only learned to continue text, so it usually carries on from your message instead of answering it. Seeing that is the clearest lesson in why fine-tuning exists.</li>
<li><strong>SFT</strong> learned turn-taking from example conversations.</li>
<li><strong>DPO</strong> is the fine-tuned model nudged towards the answers you preferred.</li>
</ul>
<p>A stage with no checkpoint yet is dimmed, and the page links to where you can train it.</p>
<h3>See how it decides</h3>
<ul>
<li><strong>X-ray</strong> shows each token of a reply as a chip. Stronger blue means the model was surer. Click a chip to see the alternatives it weighed.</li>
<li><strong>Transcript</strong> shows the raw text the model reads, special tokens included, along with how much of the context window is in use. When the window fills up, the oldest turns are dropped.</li>
<li><strong>Settings</strong> holds the system prompt, the decoding presets and sliders, and the seed. With the seed locked, the same conversation gets the same reply.</li>
</ul>
<h3>Keep and share</h3>
<ul>
<li><strong>Try again</strong> rewrites a reply with a fresh seed, <strong>Copy</strong> copies it, and <strong>New chat</strong> starts over.</li>
<li><strong>Share model…</strong> packs a checkpoint of your choice into a .zip in your Downloads folder. Inside are <code>model.safetensors</code> (the weights), <code>config.json</code>, <code>tokenizer.json</code> and a <code>README.md</code>, plus the PyTorch code needed to load them.</li>
</ul>
<p>Keep your expectations small. A model trained for minutes or hours on a phone or laptop writes plausible sentences and follows the format, but it knows few facts and can't reason. In return, you can see every part of how it works.</p>
<p>Background: <a href="#/how-models-learn-behavior">How models learn behavior</a>.</p>
`,
};
