## hpaa

### Modes

- **Generation mode** (default)  
  If `--file_eval` is **not** provided, the script generates HPAA samples and saves them to `--hpaa_folder`.

- **Evaluation mode**  
  If both `--file_eval` **and** `--detector_name` are provided, the script evaluates the specified file(s) using the selected detector.  
  Note that `--file_eval` may include multiple files produced in Generation mode, but run Evaluation mode using a **single** detector specified by `--detector_name`.

---

### Demo

Before using any detectors, make sure to update `.env` with your own API keys and install the dependencies listed in `requirements.txt`. Additionally, configure the `download_path` field in `./src/detectors.yaml` to specify where downloaded open-source models (e.g., Llama Guard 8B, ShieldGemma-2B, ShieldGemma-9B) should be stored.

Run `run.sh` to see examples for generating HPAA samples or evaluating the generated samples.

If you need to specify which GPU to use, you can set the environment variable before running the script.  
For example, `env["CUDA_VISIBLE_DEVICES"] = "0"` specifies that GPU 0 should be used.

The raw detector outputs will be saved, and these results are then aggregated to compute the final detection rate.

---

### Datasets

- **Toxic Text Dataset**  
  Location: `./data/toxic.Advbench_10.csv`

- **Benign Text Dataset**  
  Location: `./data/benign.*.csv`

- **User Study Dataset I**  
  Location: `./data/Dataset_I.csv`

- **User Study Dataset II**  
  Location: `./data/Dataset_II.csv`

---

## Ethical Considerations

### **Vulnerability Disclosure**

Our research examines adversarial prompting techniques for evaluating the robustness of language-model–based content moderation systems, including dedicated moderation APIs, general-purpose LLM APIs configured for moderation via prompting, and locally deployed LLM moderation components. All experiments were performed exclusively through publicly documented interfaces or within locally controlled environments. We did not access internal systems, circumvent protections, or interact with any user data.

The issues identified in this work represent behavioral limitations of model-based moderation mechanisms rather than vulnerabilities in deployed software or infrastructure, and therefore do not fall under traditional vulnerability-disclosure processes. No harm was caused to vendors or users at any point during our study.

---

### **Human Subjects Research**

As noted earlier, this study received IRB approval, and all procedures related to informed consent, participant protections, and data handling were reviewed and implemented accordingly. This section provides additional clarification to align with standard human-subjects research requirements.

Our study involved participant judgments on brief text excerpts—some containing sensitive language—used solely to evaluate evasion of LLM-based moderation mechanisms. Upon review, the IRB classified the study as *minimal risk*, and participants were free to withdraw at any time.

No personally identifiable information (PII) was collected. All responses were anonymized upon submission and analyzed only in aggregate form. Data were stored securely with restricted access, ensuring that participants could not be re-identified. These procedures ensure compliance with ethical standards for human-subjects research and the mitigation of any potential harms.

