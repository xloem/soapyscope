#!/usr/bin/env python3

import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

import numpy as np

# this class ended up wrapping soapy.device
# with purpose of abstracting live stream
# it could possibly be removed
class scopish:
	def __init__(self, soapy_args, rate, channel=0, bufsize=1024):
		soapy_args_list = SoapySDR.Device.enumerate(soapy_args)
		if len(soapy_args_list) > 1:
			raise Exception('more than one device matches', soapy_args, *soapy_args_list)
		self.device_args = soapy_args_list[0]
		self.channel = channel
		self.device = SoapySDR.Device(self.device_args)
		print(self.device.getGainRange(SOAPY_SDR_RX, channel))
		print(self.device.hasGainMode(SOAPY_SDR_RX, channel))
		#self.device.writeSetting('direct_samp', '2')
		self.device.writeSetting('testmode', 'true')
		self.device.setSampleRate(SOAPY_SDR_RX, self.channel, rate)

		#self.device.setFrequency(SOAPY_SDR_RX, self.channel, 100000000)
		self.device.setGainMode(SOAPY_SDR_RX, self.channel, True)
		#self.gain = #50#0#20
		#self.device.setGain(SOAPY_SDR_RX, self.channel, 0)

		self.buf = np.zeros(bufsize, np.complex64)
		self.stream_args = [SOAPY_SDR_RX, SOAPY_SDR_CF32, [self.channel]]
		self.stream = self.device.setupStream(*self.stream_args)
	def test_mode(self, enable):
		self.device.writeSetting('testmode', 'true' if enable else 'false')
	def direct_sampling(self, num):
		if not num:
			num = 0
		self.device.writeSetting('direct_samp', str(num))
	@property
	def gain(self):
		return self.device.getGain(SOAPY_SDR_RX, self.channel)
	@property
	def frequency(self):
		return self.device.getFrequency(SOAPY_SDR_RX, self.channel)
	@gain.setter
	def gain(self, gain):
		self.device.setGain(SOAPY_SDR_RX, self.channel, gain)
		print('gain', gain, '->', self.gain)
	@property
	def metadata(self):
		return dict(
			driver_key = self.device.getDriverKey(),
			hardware_key = self.device.getHardwareKey(),
			device_args = self.device_args,
			hardware_info = dict(self.device.getHardwareInfo()),
			settings = {s.key:s.value for s in self.device.getSettingInfo()},
			stream_args = self.stream_args,
			stream_args_info = {s.key:s.value for s in self.device.getStreamArgsInfo(SOAPY_SDR_RX, self.channel)}
		)
	def __enter__(self):
		self.device.activateStream(self.stream)
		return self.recv
	def __exit__(self, *params):
		self.device.closeStream(self.stream)
	def recv(self, ct=None):
		status = self.device.readStream(self.stream, [self.buf], len(self.buf), timeoutUs=int(5e6))
		if status.ret < 0:
			raise Exception(SoapySDR.errToStr(status.ret))
		return self.buf[:status.ret]#
	def start(self):
		self.running = True
		while self.running:
			data = self.recv()
	def stop(self):
		self.running = False

import sdl2.ext, os, ctypes
class display:
	def __init__(self):
		if 'DISPLAY' not in os.environ:
			os.environ['DISPLAY'] = ':0'
		sdl2.ext.init()
		self.window = sdl2.SDL_CreateWindow(b'Hello, world!', sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED, 800, 600, sdl2.SDL_WINDOW_SHOWN | sdl2.SDL_WINDOW_MAXIMIZED)
		self.renderer = sdl2.SDL_CreateRenderer(self.window, -1, sdl2.SDL_RENDERER_ACCELERATED)
		self.texture = sdl2.SDL_CreateTexture(self.renderer, sdl2.SDL_PIXELFORMAT_RGB888, sdl2.SDL_TEXTUREACCESS_STREAMING, *self.size)
	@property
	def size(self):
		w, h = ctypes.c_int(), ctypes.c_int()
		sdl2.SDL_GetRendererOutputSize(self.renderer, ctypes.byref(w), ctypes.byref(h))
		return w.value, h.value	
	def __enter__(self):
		pixels = ctypes.c_void_p()
		pitch = ctypes.c_int()
		sdl2.SDL_LockTexture(self.texture, ctypes.cast(0, ctypes.POINTER(sdl2.SDL_Rect)), ctypes.byref(pixels), ctypes.byref(pitch))
		self.pixels = ctypes.cast(pixels, ctypes.POINTER(ctypes.c_int))
		self.pitch = pitch.value // 4
	def __exit__(self, *params):
		sdl2.SDL_UnlockTexture(self.texture)
	def update(self):
		events = sdl2.ext.get_events()
		for event in events:
			if event.type == sdl2.SDL_QUIT:
				raise Exception('SDL_QUIT')
		#for x in range(self.size[1] * pitch.value // 4 // 2):
		#	pixels[x] = 0xff#ffffff
		sdl2.SDL_RenderCopy(self.renderer, self.texture, ctypes.cast(0, ctypes.POINTER(sdl2.SDL_Rect)), ctypes.cast(0, ctypes.POINTER(sdl2.SDL_Rect)))
		sdl2.SDL_RenderPresent(self.renderer)
	def __del__(self):
		sdl2.ext.quit()


import os
class dir_dataset:
	def __init__(self, name):
		self.name = name
		self.path = os.path.join(os.path.abspath('.'), name)
		os.makedirs(self.path, exist_ok=True)
	def create(self, itemname, **metadata):
		item = dir_dataset.item(self, itemname)
		item.metadata = metadata
		return item
	def load(self, itemname):
		return dir_dataset.item(self, itemname)
	class item:
		def __init__(self, dataset, name):
			self.dataset = dataset
			self.name = name
			self.prefix = os.path.join(self.dataset.path, self.name)
			self.bin_path = self.prefix + '.c64'
			self.metadata_path = self.prefix + '.json'
			self.bin_file = open(self.bin_path, 'r+b')
			self.size_bytes = self.bin_file.seek(0, 2)
			self.bin_file.seek(0, 0)
		@property
		def metadata(self):
			with open(self.metadata_path, 'rt') as metadata_file:
				return json.load(metadata_file)
		@metadata.setter
		def write_metadata(self, metadata):
			assert not os.path.exists(self.metadata_path)
			with open(self.metadata_path, 'wt') as metadata_file:
				json.dump(metadata, metadata_file)
		@property
		def length(self):
			return self.size_bytes / 8
		def write(self, data):
			assert self.bin_file.tell() == self.size_bytes
			assert data.dtype is np.complex64
			self.bin_file.write(data.data)
		def read(self, offset=0, size=None):
			if size is None:
				size_bytes = self.size_bytes
			else:
				size_bytes = size * 8
			self.bin_file.seek(offset * 8)
			return np.frombuffer(self.bin_file.read(size_bytes), dtype=np.complex64)
		def __len__(self):
			return self.length
		def __getitem__(self, idx):
			assert type(idx) is tuple
			start, end, step = idx
			assert step is None
			if start < 0:
				start = self.length + start
			if end is not None:
				if end < 0:
					end = self.length + end - start
				else:
					end -= start
			return self.read(start, end)
		def __del__(self):
			self.bin_file.close()

class Loop:
	def __init__(self, scopish, dataset):
		self.source = scopish
		self.dataset = dataset
	def start(self):
		self.running = True
		metadata = self.source.metadata
		dataitem = self.dataset.create(f'{self.source.channel}-{self.frequency}-{time.time()}', **metadata)
		while self.running:
			print('WARNING: expecting dropped data in undesigned recv loop')
			data = self.source.recv()
			dataitem.write(data)
	def stop(self):
		self.running = False

import signal
class sigeventpair:
	def __init__(self, loop, start_signum = signal.SIGUSR1, stop_signum = signal.SIGUSR2):
		self.start_signum = start_signum
		self.stop_signum = stop_signum
		self.loop = loop
	def __enter__(self):
		self._old_start = signal.signal(self.start_signum, self.on_start)
		self._old_stop = signal.signal(self.stop_signum, self.on_stop)
	def __exit__(self, *params):
		signal.signal(self.start_signum, self._old_start)
		signal.signal(self.stop_signum, self._old_stop)
	def on_start(self, signum, frame):
		self.loop.start()
	def on_stop(self, signum, frame):
		self.loop.stop()
		

if __name__ == '__main__':
	#ch = display()
	#while True:
	#	with ch:
	#		ch.pixels[16] = 0x00FF00
	#	ch.update()
	scop = scopish('driver=rtlsdr,testmode=true', rate=2048000, bufsize=1024*1024*64)
	scop.test_mode = True
	#last = None
	last_num = 0
	with scop as recv:
		while True:
			data = recv().view(dtype=np.float32)
			data = (data * 128 + 127.4).astype(int)
			dropped = (data[1:] - data[:-1]).max() - 1
			print(dropped, 'dropped mid-chunk')
			print(data[0] - last_num - 1, 'dropped inter-chunk')
			last_num = data[-1]
	#		val = data.mean()
	#		if val != last:
	#			print(val)
	#			last = val
	#		#if data.max() == 0:
	#		#	scop.gain += 1