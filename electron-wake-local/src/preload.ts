import { contextBridge, ipcRenderer } from "electron";

export type ModelPaths = {
  modelsDir: string;
  melspectrogram: string;
  embedding: string;
  sileroVad: string;
  wakeword: string;
  wasmDir: string;
};

contextBridge.exposeInMainWorld("wakeBridge", {
  getModelPaths: (): Promise<ModelPaths> => ipcRenderer.invoke("wake:get-model-paths"),
  onStopTts: (cb: () => void) => {
    ipcRenderer.on("wake:stop-tts", cb);
    return () => ipcRenderer.removeListener("wake:stop-tts", cb);
  },
  notifyWake: () => ipcRenderer.send("wake:detected"),
  notifyBargeIn: () => ipcRenderer.send("wake:barge-in"),
});
