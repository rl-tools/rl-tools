import assert from "node:assert/strict";
import {test} from "node:test";
import {create_touch_stick} from "../../../static/hyperdrone/touch-stick.js";

function pad(){
    const element = new EventTarget();
    const style = new Map();
    const thumb = {
        style: {setProperty: (name, value) => style.set(name, value)},
        getBoundingClientRect: () => ({width: 20, height: 20}),
    };
    const captures = new Set();
    const releases = [];
    let bounds = {left: 10, top: 20, width: 120, height: 120};
    let reads = 0;
    element.querySelector = selector => {
        assert.equal(selector, ".stick-thumb");
        return thumb;
    };
    element.getBoundingClientRect = () => {
        reads++;
        return bounds;
    };
    element.setPointerCapture = id => captures.add(id);
    element.hasPointerCapture = id => captures.has(id);
    element.releasePointerCapture = id => {
        captures.delete(id);
        releases.push(id);
        send("lostpointercapture", {pointerId: id});
    };
    function send(type, fields = {}){
        const event = new Event(type, {cancelable: true});
        Object.assign(event, {pointerId: 1, pointerType: "touch", button: 0, clientX: 70, clientY: 80, ...fields});
        element.dispatchEvent(event);
        return event;
    }
    return {element, style, captures, releases, send, move_bounds: value => { bounds = value; }, get reads(){ return reads; }};
}

function close(actual, expected){
    assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
}

test("stick has a neutral center, rescaled dead zone, and circular saturation", () => {
    const fixture = pad();
    const stick = create_touch_stick(fixture.element, () => true);
    fixture.send("pointerdown");
    assert.deepEqual([stick.x, stick.y], [0, 0]);
    fixture.send("pointermove", {clientX: 74});
    assert.deepEqual([stick.x, stick.y], [0, 0]);
    fixture.send("pointermove", {clientX: 97.5});
    close(stick.x, 0.5);
    close(stick.y, 0);
    fixture.send("pointermove", {clientX: 120, clientY: 130});
    close(stick.x, Math.SQRT1_2);
    close(stick.y, Math.SQRT1_2);
    close(Math.hypot(stick.x, stick.y), 1);
    close(Math.hypot(parseFloat(fixture.style.get("--stick-x")), parseFloat(fixture.style.get("--stick-y"))), 50);
    fixture.send("pointermove", {clientX: -500, clientY: 80});
    close(stick.x, -1);
    close(stick.y, 0);
});

test("disabled pads, mouse input, and non-primary buttons do not claim pointers", () => {
    const fixture = pad();
    let enabled = false;
    const stick = create_touch_stick(fixture.element, () => enabled);
    assert.equal(fixture.send("pointerdown", {clientX: 120}).defaultPrevented, false);
    enabled = true;
    fixture.send("pointerdown", {pointerType: "mouse", clientX: 120});
    fixture.send("pointerdown", {pointerType: "pen", button: 2, clientX: 120});
    assert.deepEqual([stick.x, stick.y], [0, 0]);
    assert.equal(fixture.captures.size, 0);
    assert.equal(fixture.reads, 0);
    assert.equal(fixture.send("pointerdown", {pointerType: "pen", clientX: 120}).defaultPrevented, true);
    assert.equal(stick.x, 1);
});

test("extra fingers cannot steal or release a stick and owned motion remains captured outside the pad", () => {
    const fixture = pad();
    const stick = create_touch_stick(fixture.element, () => true);
    fixture.send("pointerdown", {clientX: 120});
    fixture.send("pointerdown", {pointerId: 2, clientX: 20});
    fixture.send("pointermove", {pointerId: 2, clientX: 20});
    fixture.send("pointerup", {pointerId: 2});
    fixture.send("pointercancel", {pointerId: 2});
    fixture.send("lostpointercapture", {pointerId: 2});
    assert.equal(stick.x, 1);
    assert.deepEqual([...fixture.captures], [1]);
    fixture.send("pointermove", {clientX: 70, clientY: -100});
    assert.equal(stick.y, -1);
    fixture.send("pointerup");
    assert.deepEqual([stick.x, stick.y], [0, 0]);
    assert.equal(fixture.captures.size, 0);
    assert.deepEqual(fixture.releases, [1]);
});

test("two pads accept simultaneous fingers even when the second is not the primary pointer", () => {
    const left_pad = pad();
    const right_pad = pad();
    const left = create_touch_stick(left_pad.element, () => true);
    const right = create_touch_stick(right_pad.element, () => true);
    left_pad.send("pointerdown", {pointerId: 1, isPrimary: true, clientX: 120});
    right_pad.send("pointerdown", {pointerId: 2, isPrimary: false, clientY: 30});
    assert.deepEqual([left.x, left.y, right.x, right.y], [1, 0, 0, -1]);
    left_pad.send("pointerup", {pointerId: 1});
    assert.deepEqual([left.x, left.y, right.x, right.y], [0, 0, 0, -1]);
    right_pad.send("pointerup", {pointerId: 2});
    assert.deepEqual([right.x, right.y], [0, 0]);
});

for(const type of ["pointerup", "pointercancel", "lostpointercapture"]){
    test(`${type} centers the stick and permits a new pointer`, () => {
        const fixture = pad();
        const stick = create_touch_stick(fixture.element, () => true);
        fixture.send("pointerdown", {clientX: 120});
        if(type === "lostpointercapture"){
            fixture.captures.delete(1);
        }
        fixture.send(type);
        assert.deepEqual([stick.x, stick.y], [0, 0]);
        assert.equal(fixture.style.get("--stick-x"), "0px");
        assert.equal(fixture.style.get("--stick-y"), "0px");
        fixture.send("pointerdown", {pointerId: 2, clientX: 20});
        assert.equal(stick.x, -1);
    });
}

test("clear is idempotent when releasing capture synchronously dispatches lostpointercapture", () => {
    const fixture = pad();
    const stick = create_touch_stick(fixture.element, () => true);
    fixture.send("pointerdown", {clientX: 120});
    stick.clear();
    stick.clear();
    assert.deepEqual([stick.x, stick.y], [0, 0]);
    assert.deepEqual(fixture.releases, [1]);
    fixture.send("pointermove", {clientX: 20});
    assert.deepEqual([stick.x, stick.y], [0, 0]);
});

test("disabling a held stick releases capture before processing more motion", () => {
    const fixture = pad();
    let enabled = true;
    const stick = create_touch_stick(fixture.element, () => enabled);
    fixture.send("pointerdown", {clientX: 120});
    enabled = false;
    fixture.send("pointermove", {clientX: 20});
    assert.deepEqual([stick.x, stick.y], [0, 0]);
    assert.equal(fixture.captures.size, 0);
});

test("pad geometry is measured only at capture, then refreshed for the next touch", () => {
    const fixture = pad();
    const stick = create_touch_stick(fixture.element, () => true);
    fixture.send("pointerdown");
    fixture.move_bounds({left: 110, top: 20, width: 120, height: 120});
    fixture.send("pointermove", {clientX: 120});
    assert.equal(stick.x, 1);
    assert.equal(fixture.reads, 1);
    fixture.send("pointerup");
    fixture.send("pointerdown", {clientX: 170});
    assert.deepEqual([stick.x, stick.y], [0, 0]);
    assert.equal(fixture.reads, 2);
});

test("only the pad suppresses its context menu", () => {
    const fixture = pad();
    create_touch_stick(fixture.element, () => true);
    assert.equal(fixture.send("contextmenu").defaultPrevented, true);
});
