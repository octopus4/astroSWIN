#feature-id   AstroSWIN
#feature-info AstroSWIN processor for enhancement and sharpening

#include <pjsr/Sizer.jsh>
#include <pjsr/NumericControl.jsh>

var pxTempDirPath = "pixinsight_tmp";
var pxTempInputName = "pixinsight_tmp_input.tiff";
var pxTempOutputName = "pixinsight_tmp_output.tiff";
var defaultSampleBits = 16;

var configAppProperty = "APP_PATH";
var configModelsProperty = "MODELS";

var aswinPath = File.homeDirectory + "/AppData/Local/aswin";

function AstroSWINEngine(config, params) {
    // many thanks to GraXpert opensourced code
    this.process = new ExternalProcess();
    this.config = config;
    this.blendingParamToKey = {
        " --patch-size ": "patchSize",
        " --beta ": "maskBeta",
        " --const ": "maskConst",
        " --mul ": "maskMul"
    };
    this.params = params;

    this.execute = function (inputPath, outputPath) {
        let appPath = aswinPath + "/" + this.config[configAppProperty];
        let modelPath = aswinPath + "/" + this.params["selectedModel"];

        command = appPath;
        command += " -i " + inputPath;
        command += " -o " + outputPath;
        command += " -m " + modelPath;

        if (this.params["doBlending"]) {
            let blendingParamNames = Object.keys(this.blendingParamToKey);
            for (let i = 0; i < blendingParamNames.length; ++i) {
                let paramName = blendingParamNames[i];
                let paramKey = this.blendingParamToKey[paramName];
                command += paramName + this.params[paramKey];
            }
        }
        Console.writeln(command);

        this.process.start(command);
        if (!this.process.waitForStarted(10000)) {
            throw "Failed to start process";
        }

        var process = this.process;
        this.process.onStandardOutputDataAvailable = function () {
            let outputLines = process.standardOutput.toString().split("\r\n");
            if (outputLines.length > 1) {
                Console.writeln(outputLines[0]);
            }
            Console.flush();
            processEvents();
        }

        while (this.process.isRunning) {
            processEvents();
        }
        if (this.process.exitCode != 0) {
            Console.writeln(this.process.error);
            throw this.process.error;
        }
    }
}

function HFProcessorDialog() {
    this.__base__ = Dialog;
    this.__base__();

    this.windowTitle = "AstroSWIN";
    this.minHeight = 250;
    this.maxHeight = 250;
    this.minWidth = 300;
    this.maxWidth = 300;
    this.processingStarted = false;

    var windows = ImageWindow.windows;
    var config = JSON.parse(File.readTextFile(aswinPath + "/config.json"));

    var viewsComboBox = new ComboBox(this);
    viewsComboBox.addItem("Select Image");
    for (let i = 0; i < windows.length; ++i) {
        viewsComboBox.addItem(windows[i].mainView.id);
    }
    viewsComboBox.currentItem = 0;

    var modelsComboBox = new ComboBox(this);
    modelsComboBox.addItem("Select Model");
    for (let i = 0; i < config[configModelsProperty].length; ++i) {
        modelsComboBox.addItem(config[configModelsProperty][i]);
    }
    modelsComboBox.currentItem = 0;

    var patchSizeSelector = new NumericControl(this);
    patchSizeSelector.setReal(false);
    patchSizeSelector.setRange(8, 64);
    patchSizeSelector.setValue(32);
    patchSizeSelector.text = "Blending patch size";

    var maskBetaSelector = new NumericControl(this);
    maskBetaSelector.setReal(true);
    maskBetaSelector.setRange(0, 1);
    maskBetaSelector.setPrecision(3);
    maskBetaSelector.setValue(0.05);
    maskBetaSelector.text = "Blending mask beta param";

    var maskConstSelector = new NumericControl(this);
    maskConstSelector.setReal(true);
    maskConstSelector.setRange(0, 1);
    maskConstSelector.setPrecision(3);
    maskConstSelector.setValue(0);
    maskConstSelector.text = "Blending mask const param";

    var maskMulSelector = new NumericControl(this);
    maskMulSelector.setReal(true);
    maskMulSelector.setRange(0, 1);
    maskMulSelector.setPrecision(3);
    maskMulSelector.setValue(1);
    maskMulSelector.text = "Blending mask multiplication param";

    var blendingControls = [
        patchSizeSelector,
        maskBetaSelector,
        maskConstSelector,
        maskMulSelector
    ];
    var blendingCheckBox = new CheckBox(this);
    blendingCheckBox.text = "Perform auto-blending after processing";
    blendingCheckBox.onCheck = function () {
        for (let i = 0; i < blendingControls.length; ++i) {
            blendingControls[i].enabled = this.checked;
        }
    }
    blendingCheckBox.onCheck();

    this.processButton = new PushButton(this);
    this.processButton.text = "Process";
    this.processButton.onClick = function processImage() {
        if (this.processingStarted) {
            return;
        }
        this.processingStarted = true;

        let currentItemIndex = viewsComboBox.currentItem - 1;
        if (currentItemIndex < 0) {
            new MessageBox("No view selected!", "Error").execute();
            return;
        }

        var selectedWindow = ImageWindow.windows[currentItemIndex];
        selectedWindow.setSampleFormat(defaultSampleBits, floatSample = false);

        try {
            Console.show();
            Console.writeln("Starting processing...");

            let tmpPath = File.systemTempDirectory + "/" + pxTempDirPath;
            let inputPath = tmpPath + "/" + pxTempInputName;
            let outputPath = tmpPath + "/" + pxTempOutputName;
            if (!File.directoryExists(tmpPath)) {
                File.createDirectory(tmpPath);
                Console.writeln("Created tmp dir");
            }

            selectedWindow.saveAs(inputPath, false, false, false, false);
            Console.writeln("Image saved to:", inputPath);

            Console.writeln("Executing astro swin");
            let params = {
                selectedModel: config[configModelsProperty][modelsComboBox.currentItem - 1],
                doBlending: blendingCheckBox.checked,
                patchSize: patchSizeSelector.value,
                maskBeta: maskBetaSelector.value,
                maskConst: maskConstSelector.value,
                maskMul: maskMulSelector.value
            };
            Console.writeln(JSON.stringify(params));
            let astroSwin = new AstroSWINEngine(config, params);
            astroSwin.execute(inputPath, outputPath);
            Console.writeln("External processing completed");

            let newId = selectedWindow.mainView.id + "_processed";
            let processed = ImageWindow.open(outputPath, newId, '', false)[0];
            processed.show();
            processed.bringToFront();

            File.remove(inputPath);
            File.remove(outputPath);
            Console.writeln("Done!");
        } catch (error) {
            Console.writeln(error);
            new MessageBox("Processing error", "Error").execute();
        }
        this.processingStarted = false;
    };

    this.sizer = new VerticalSizer(this);
    this.sizer.spacing = 8;
    this.sizer.margin = 8;

    this.sizer.add(viewsComboBox);
    this.sizer.add(modelsComboBox);
    this.sizer.add(blendingCheckBox);
    this.sizer.add(patchSizeSelector);
    this.sizer.add(maskBetaSelector);
    this.sizer.add(maskConstSelector);
    this.sizer.add(maskMulSelector);
    this.sizer.add(this.processButton);

    this.adjustToContents();
}

HFProcessorDialog.prototype = new Dialog;

function main() {
    var dialog = new HFProcessorDialog();
    dialog.execute();
}

main();
