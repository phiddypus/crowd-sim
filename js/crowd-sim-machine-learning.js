/*
This project was made from scratch as a demonstration of technical ability.
This file took in X hours.
Learning Two.js was included in the process and time.

UNFINISHED
The goal is to use the crowd sim script and add machine learning functionality to each agent, which will be stored in each Agent's Brain.
*/

const
    TWO_PI = 2 * Math.PI,
    NUM_AGENTS = 100,
    SHOW_VISION = true,
    AGENT_CONFIG = {
        size: {
            min: 4.5, range: 1
        },
        fov: {
            min: Math.PI / 8, range: 3 * Math.PI / 8
        },
        dist: {
            min: 90, range: 20
        }
    },
    EPSILON = {
        c: 5, // cartesian
        r: 0.003 // radial
    },
    MAX_VEL = {
        c: Math.sqrt(2),
        r: Math.PI / 30
    },
    MAX_ACC = {
        c: MAX_VEL.c,
        r: MAX_VEL.r / 20
    }

let cnv = new Two({
    type: Two.Types.canvas,
    fullscreen: true,
    autostart: true
}).appendTo(document.body)

class Angle {
    static normalize (r) {
        let n = r % TWO_PI
        return n + (n < 0 ? TWO_PI : 0)
    }

    static difference (r0, r1) {return (r0 - r1 + 3 * Math.PI) % TWO_PI - Math.PI}

    static equals (r0, r1) {return Math.abs(Angle.difference(r0, r1)) < EPSILON.r}
}

class RVector {
    constructor (x = 0, y = 0, r = 0) {Object.assign(this, {x, y, r})}

    static fromMH (m = 0, h = 0, r = 0) {return new RVector(        
        Math.cos(h) * m,
        Math.sin(h) * m,
        r
    )}
    
    set (x = this.x, y = this.y, r = this.r) {
        this.x = x
        this.y = y
        this.r = r
    }

    getMagnitude () {return Math.hypot(this.y, this.x)}

    setMagnitude (magnitude) {
        let thisMagnitude = this.getMagnitude()
        if (magnitude && thisMagnitude) {
            let f = magnitude / thisMagnitude
            this.x *= f
            this.y *= f
        } else {
            this.x = magnitude
            if (!magnitude) this.y = 0
        }
    }

    getHeading () {return Angle.normalize(Math.atan2(this.y, this.x))}

    setHeading (heading) {this.setMagHead(this.getMagnitude(), heading)}

    setMagHead (magnitude, heading) {
        this.x = Math.cos(heading) * magnitude
        this.y = Math.sin(heading) * magnitude
    }

    add (oRVector, useR = true) {return new RVector(this.x + oRVector.x, this.y + oRVector.y, useR ? this.r + oRVector.r : undefined)}

    subtract (oRVector, useR = true) {return new RVector(this.x - oRVector.x, this.y - oRVector.y, useR ? this.r - oRVector.r : undefined)}

    multiply (scalar, scalarR = 1) {return new RVector(this.x * scalar, this.y * scalar, this.r * scalarR)}

    equals (oRVector, useR = true) {return (
        (!useR || Angle.equals(this.r, oRVector.r)) &&
        this.subtract(oRVector).getMagnitude() < EPSILON.c
    )}
}

class PhysPoint {
    constructor (pos = new RVector(), vel = new RVector(), acc = new RVector()) {Object.assign(this, {pos, vel, acc})}

    update () {
        this.vel = this.vel.add(this.acc)
        this.pos = this.pos.add(this.vel)
        this.pos.r = Angle.normalize(this.pos.r)

        let magnitude = this.vel.getMagnitude()
        this.vel.setMagnitude(magnitude > MAX_VEL.c ? MAX_VEL.c : magnitude)
        this.vel.r = Math.min(Math.max(this.vel.r, -MAX_VEL.r), MAX_VEL.r)
    }
}

class Agent extends PhysPoint {
    static idCount = -1

    static calcAngle (weights) {return weights.map(w => w.h * w.m).reduce((a, n) => a + n, 0) / weights.map(w => w.m).reduce((a, n) => a + n, 0)}

    static random (cnv, config) {
        let r = Math.random(), n = Math.random(), t = [0, 0]
        if (n <= 0.5) {
            t[0] = (r * 0.8 + 0.1) * cnv.width
            t[1] = n <= 0.25 ? cnv.height * 0.1 : cnv.height * 0.9
        } else {
            t[1] = (r * 0.8 + 0.1) * cnv.height
            t[0] = n <= 0.75 ? cnv.width * 0.1 : cnv.width * 0.9
        }

        return new Agent(cnv,
            new RVector(Math.random() * cnv.width, Math.random() * cnv.height, Math.random() * TWO_PI),
            new RVector(...t),
            Math.random() * config.size.range + config.size.min,
            {
                fov: Math.random() * config.fov.range + config.fov.min,
                dist: Math.random() * config.dist.range + config.dist.min
            }
        )
    }
    
    constructor (cnv, pos, dest, size, {fov, dist}, vel = new RVector(), acc = new RVector(), fwd = 0.1) {
        super(pos, vel, acc)
        this.dest = dest; this.size = size; this.fwd = fwd
        this.sight = {fov, dist}

        this.brain = new Brain("inputs")

        this.render = cnv.makeGroup(
            cnv.makeCircle(0, 0, this.size),
            SHOW_VISION ?
                cnv.makeArcSegment(0, 0, this.size, this.sight.dist, -this.sight.fov, this.sight.fov) :
                cnv.makeLine(this.size, 0, 2 * this.size, 0)
        )
        this.render.fill = 'rgb(0,0,0,0)'
        this.render.position.set(this.pos.x, this.pos.y)
        this.render.rotation = this.pos.r
        this.id = Agent.idCount += 1
    }

    sees (oAgent) { if (oAgent) {
        let difv = oAgent.pos.subtract(this.pos, false)
        let dist = difv.getMagnitude()

        return (
            dist - oAgent.size <= this.sight.dist && // within distance?
            Math.abs(Angle.difference(difv.getHeading(), this.pos.r)) <= this.sight.fov + Math.atan2(oAgent.size, dist) // within angle?
        ) ? difv : false
    }}

    visible (oAgents) { //unoptimized
        return oAgents.map((oAgent) => this.sees(oAgent)).filter((x) => x)
    }

    update (visible) {
        let vto = this.dest.subtract(this.pos, false)

        if (vto.equals(new RVector(), false)) return 'delete'
        
        [this.acc.v, this.acc.r] = this.brain.run("input")

        super.update()
        this.draw()
    }

    draw () {
        this.render.position.set(this.pos.x, this.pos.y)
        this.render.rotation = this.pos.r
    }
}

class Brain {
    constructor (inputs) {

    }
}

// function main() {
    let agents = []
    let paused = false
    document.addEventListener('click', (e) => {paused = !paused})

    for (let i = 0; i < NUM_AGENTS; i++) agents[i] = Agent.random(cnv, AGENT_CONFIG)

    cnv.bind('update', () => { if (!paused)
        agents.forEach((a, i, agents) => {
            if (a?.update(a.visible(agents.toSpliced(i, 1))) == 'delete') delete agents[i]
        })
    })
// }

// main()

// let a = new Agent(cnv,
//     new RVector(100, 100, Math.PI),
//     new RVector(500, 500),
//     10, {fov: Math.PI / 4, dist: 100}
// )

// let b = new Agent(cnv,
//     new RVector(399, 399),
//     new RVector(500, 500),
//     10, {fov: Math.PI / 8, dist: 100}
// )

// let mousea = new Agent(cnv,
//     new RVector(0, 0, Math.PI),
//     new RVector(500, 500),
//     10, {fov: Math.PI / 4, dist: 100}
// )
// document.addEventListener('mousemove', MAUP)
// function MAUP (e) {
//     if (!paused) mousea.pos.set(e.clientX, e.clientY)
// }

// cnv.makeCircle(500, 500, 2)


// let agents = [a, b, mousea]

// var paused = false
// document.addEventListener('click', (e) => {paused = !paused})

// function u () {
//     if (!paused) {
//         agents.forEach((agent, i, agents) => {
//             agent.update(agent.visible(agents.toSpliced(i, 1)))
//         })
//     }
// }

// cnv.bind('update', u)