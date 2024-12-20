let canvas = document.getElementById("drawingCanvas");
let prediction = document.getElementById("predictButton")
console.log("Hello")
console.log(canvas)

class drawingCanvas{
    constructor(canvas){
        this.canvas = canvas
        this.mode = this.canvas.getContext("2d")
        this.isdrawing = false
        this.height = this.canvas.height;
        this.width = this.canvas.width;
        this.onmousedrag = this.onmousedrag.bind(this)
        this.onmouseleave = this.onmouseleave.bind(this)
        this.onmousedown = this.onmousedown.bind(this)
        this.onmouseup = this.onmouseup.bind(this)
        this.canvas.addEventListener("mousedown", this.onmousedown)
        this.canvas.addEventListener("mouseup", this.onmouseup)
        this.canvas.addEventListener("mouseleave", this.onmouseleave)
        this.canvas.addEventListener("mousemove", this.onmousedrag)
        this.mode.fillStyle = "black"
        this.mode.fillRect(0, 0, this.width, this.height)
    }

    onmousedown(message){
        //console.log("downs")
        this.isdrawing = true
        //let mouseposition = [message.clientX, message.clientY]
        let mouseposition = [message.clientX, message.clientY]
        positions.push([mouseposition])
    }

    onmouseup(message){
        this.isdrawing = false
    }

    onmousedrag(message){
        if (this.isdrawing == true){
            let mouseposition = [message.clientX, message.clientY]
            positions [positions.length - 1].push(mouseposition)
            this.draw()
        }
    }

    onmouseleave(message){
        this.isdrawing = false
    }

    draw(){
        this.mode.beginPath()
        this.mode.strokeStyle = "white"
        this.mode.lineWidth = 10
        for(let line = 0; line < positions.length; line++){
            this.connectLines(positions[line])
        }
    }

    connectLines(singleLine){
        console.log(singleLine)
        for(let line = 0; line < singleLine.length -1; line++){
            let startPos = this.relative_position(singleLine[line])
            let endPos = this.relative_positioPn(singleLine[line+1])
            this.mode.moveTo(startPos[0], startPos[1])
            this.mode.lineTo(endPos[0], endPos[1])
            this.mode.stroke()
        }
    }

    relative_position(ev) {
        let bb = this.canvas.getBoundingClientRect();
        const scaleX =1 //this.canvas.width / bb.width;
        const scaleY = 1//sthis.canvas.height / bb.height;
        let xPos = ev.clientX;
        let yPos = ev.clientY;
        let relPos = [(xPos - bb.left) * scaleX, (yPos - bb.top) * scaleY];
        return relPos;
      }
}

function sendData(){
    fetch("/mnist_playground", {
        method: "POST",
        body: JSON.stringify(positions),
        headers: {"Content-type": "application/json; charset=UTF-8"},
    }).then((response) => response.json());
}

let positions = [];
prediction.addEventListener("click", sendData)
let dm = new drawingCanvas(canvas);
/*
function print(message){
    console.log("Clicked")
    console.log(message)
}
function print1(message){
    console.log("Mouse released.")
    console.log(message)

}

canvas.addEventListener("mousedown", print);
canvas.addEventListener("mouseup", print1);
*/