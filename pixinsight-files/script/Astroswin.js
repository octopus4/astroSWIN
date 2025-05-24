#feature-id   AstroSWIN
#feature-info AstroSWIN processor for enhancement and sharpening

#include <pjsr/Sizer.jsh>

var pxTempDirPath = "pixinsight_tmp";
var pxTempInputName = "pixinsight_tmp_input.tiff";
var pxTempOutputName = "pixinsight_tmp_output.tiff";
var defaultSampleBits = 16;

var aswinPath = File.homeDirectory + "/AppData/Local/aswin";

function AstroSWINEngine() {
    // many thanks to GraXpert opensourced code

    this.process = new ExternalProcess();
    this.config = JSON.parse(File.readTextFile(aswinPath + "/config.json"));

    this.execute = function (inputPath, outputPath) {
        let appPath = aswinPath + "/" + this.config["APP_PATH"];
        let modelPath = aswinPath + "/" + this.config["MODEL_PATH"];

        command = appPath;
        command += " -i " + inputPath;
        command += " -o " + outputPath;
        command += " -m " + modelPath;

        this.process.start(command);
        if (!this.process.waitForStarted(10000)) {
            throw "Failed to start process";
        }

        var process = this.process;
        var totalOutput = "";
        this.process.onStandardOutputDataAvailable = function () {
            let outputLines = process.standardOutput.toString().split("\r\n");
            Console.writeln(outputLines[0]);
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
    this.minHeight = 50;
    this.maxHeight = 50;
    this.minWidth = 400;
    this.maxWidth = 400;
    this.processingStarted = false;

    var comboBox = new ComboBox(this);
    var windows = ImageWindow.windows;

    comboBox.addItem("Select Image");
    for (let i = 0; i < windows.length; ++i) {
        comboBox.addItem(windows[i].mainView.id);
    }
    comboBox.currentItem = 0;

    this.processButton = new PushButton(this);
    this.processButton.text = "Process";
    this.processButton.onClick = function processImage() {
        if (this.processingStarted) {
            return;
        }
        this.processingStarted = true;

        let currentItemIndex = comboBox.currentItem - 1;
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

            selectedWindow.saveAs(inputPath, queryOptions=false);
            Console.writeln("Image saved to:", inputPath);

            Console.writeln("Executing astro swin");
            let astroSwin = new AstroSWINEngine();
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

    this.sizer = new HorizontalSizer(this);
    this.sizer.spacing = 8;
    this.sizer.margin = 8;

    this.sizer.add(comboBox);
    this.sizer.addSpacing(10);
    this.sizer.add(this.processButton);

    this.adjustToContents();
}

HFProcessorDialog.prototype = new Dialog;

function main() {
    var dialog = new HFProcessorDialog();
    dialog.execute();
}

main();
