#!/usr/bin/env python3

import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

import time
import json
import numpy as np

# this class ended up wrapping soapy.device
# with purpose of abstracting live stream
# it could possibly be removed
class scopish:
	def __init__(self, soapy_args, rate, channel=0, bufsize=1024):
		soapy_args_list = SoapySDR.Device.enumerate(soapy_args)
		if len(soapy_args_list) > 1:
			raise Exception('more than one device matches', soapy_args, *soapy_args_list)
		self.device_args = {**soapy_args_list[0]}
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
		for DISPLAY in (None, ':0', ':1'):
			try:
				if DISPLAY is not None:
					os.environ['DISPLAY'] = DISPLAY
				sdl2.ext.init()
				break
			except:
				pass
		self.window = sdl2.SDL_CreateWindow(b'Hello, world!', sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED, 540, 405, sdl2.SDL_WINDOW_RESIZABLE | sdl2.SDL_WINDOW_SHOWN | sdl2.SDL_WINDOW_MAXIMIZED)
		self.renderer = sdl2.SDL_CreateRenderer(self.window, -1, sdl2.SDL_RENDERER_ACCELERATED)
		sdl2.SDL_SetRenderDrawColor(self.renderer, 0, 255, 0, 0);
		sdl2.SDL_RenderClear(self.renderer);
		self.tex_width, self.tex_height = self.size
		self.texture = sdl2.SDL_CreateTexture(self.renderer, sdl2.SDL_PIXELFORMAT_RGB888, sdl2.SDL_TEXTUREACCESS_STREAMING, self.tex_width, self.tex_height)
		if not self.texture:
			raise Exception(sdl2.SDL_GetError())
		sdl2.SDL_SetTextureBlendMode(self.texture, sdl2.SDL_BLENDMODE_NONE)
		self.pixels = None
		self.max_tex_length = 16384
		self.max_tex_height = 16384
	@property
	def size(self):
		w, h = ctypes.c_int(), ctypes.c_int()
		sdl2.SDL_GetRendererOutputSize(self.renderer, ctypes.byref(w), ctypes.byref(h))
		return w.value, h.value	
	def __enter__(self):
		self.lockpixels()
	def lockpixels(self):
		assert self.pixels is None
		pixels = ctypes.c_void_p()
		pitch = ctypes.c_int()
		result = sdl2.SDL_LockTexture(self.texture, ctypes.cast(0, ctypes.POINTER(sdl2.SDL_Rect)), ctypes.byref(pixels), ctypes.byref(pitch))
		if result != 0:
			raise Exception(sdl2.SDL_GetError())
		self.pitch = pitch.value // 4
		self.pixels = np.ctypeslib.as_array(ctypes.cast(pixels, ctypes.POINTER(ctypes.c_uint)), shape=(self.pitch * self.tex_height,))
	def __exit__(self, *params):
		self.unlockpixels()
	def unlockpixels(self):
		sdl2.SDL_UnlockTexture(self.texture)
		self.pixels = None
	def update(self):
		events = sdl2.ext.get_events()
		for event in events:
			if event.type == sdl2.SDL_QUIT:
				raise Exception('SDL_QUIT')
			elif event.type == sdl2.SDL_MOUSEMOTION:
				motion = event.motion
				print(motion.x, motion.y)
		#for x in range(self.size[1] * pitch.value // 4 // 2):
		#	pixels[x] = 0xff#ffffff
		sdl2.SDL_RenderPresent(self.renderer)
	def render(self, x = 0, y = 0, w = None, h = None):
		if w is None or h is None:
			size = self.size
		o_rect = sdl2.SDL_Rect(x, y, size[0] if w is None else w, size[1] if h is None else h)
		result = sdl2.SDL_RenderCopy(self.renderer, self.texture, ctypes.cast(0, ctypes.POINTER(sdl2.SDL_Rect)), o_rect)#ctypes.cast(0, ctypes.POINTER(sdl2.SDL_Rect)))
		if result != 0:
			raise Exception(sdl2.SDL_GetError())
	def __del__(self):
		sdl2.ext.quit()

	def draw_dataset(self, dataset):
		width, height = self.size
		itemnames = dataset.list()[:height]
		maxlength = dataset.maxlength
		texlength = min(maxlength, self.max_tex_length)
		texheight = min(len(itemnames), self.max_tex_height)
		if self.tex_width != texlength or self.tex_height != texheight or (self.tex_width < maxlength and self.tex_height > 1):
			while True:
				try:
					self.tex_width = texlength
					self.tex_height = texheight
					if self.tex_width < maxlength and self.tex_height > 1:
						self.tex_height = 1
					self.texture = sdl2.SDL_CreateTexture(self.renderer, sdl2.SDL_PIXELFORMAT_RGB888, sdl2.SDL_TEXTUREACCESS_STREAMING, self.tex_width, self.tex_height)
					if not self.texture:
						raise Exception(sdl2.SDL_GetError())
					sdl2.SDL_SetTextureBlendMode(self.texture, sdl2.SDL_BLENDMODE_NONE)
					break
				except:
					if self.tex_height > 1:
						self.tex_height //= 2
						self.max_text_height = self.tex_height
						texheight = self.tex_height
						continue
					if self.tex_width == self.max_tex_length:
						self.max_tex_length //= 2
						self.tex_width = self.max_tex_length
						texlength = self.max_tex_length
						continue
					raise
		# texture is of texlength x texheight
		# first 1d is tiled by texlength for each row
		# then 2d is tiled by texheight using groups of rows
		#todotodo below here

		'''
		when we run it we usually encounter an error
		due to [memorydisruptionish] issues during development.
		parts and assumptions that were put one place,
		but not included in another place.
		an approach could be to look for the assumption,
		and the place needing it included, and add it
		to the comments.
		'''


		# write all texs of pixels
		# ensure the below increments both rows and columns 
		# loop over it until the image is complete
		item = None
		itemrow = 0
		itemcol = 0
		screenrow = 0
		screencol = 0
		while itemrow < len(itemnames):
			# write tex of pixels
			# if there are multiple rows, write other rows
			texrow = 0
			pxoff = 0

			with self:
				startitemrow = itemrow
				startitemcol = itemcol
				while itemrow < len(itemnames) and texrow < self.tex_height:
					# load item for texrow. this is conditional on the row changing
					if item is None:
						item = dataset.load(itemnames[itemrow])
						itemrow += 1
						itemcol = 0
						screencol = 0

					# write subrow of pixels
					c64data = (item.read(offset = itemcol, size = self.tex_width) * 128 + complex(127.4, 127.4)).round()
					# needs pxoff initialised for the tex
					# whenever self.pixels is used, it must be locked [for the tex]
					# this needs unsigned values. the values were signed
					self.pixels[pxoff:pxoff + len(c64data)] = c64data.real + c64data.imag * 0x10000 # type: wrote image instead of imag. unnatural error.
					#print(len(c64data))
					if len(c64data) < self.tex_width:
						# item over
						texrow += 1
						pxoff += self.pitch
						item = None
					else:
						# more of this row and item
						itemcol += self.tex_width
						break

			# blit tex of pixels
			nextscreenrow = int(itemrow * height / len(itemnames))
			nextscreencol = int(itemcol * width / maxlength)
			#print('orect:', screencol, screenrow, '->', nextscreencol, nextscreenrow)
			self.render(screencol, screenrow, nextscreencol - screencol, nextscreenrow - screenrow) # when startrow changed to row, this line wasn't included
			#sdl2.SDL_RenderPresent(self.renderer)
			screenrow = nextscreenrow
			screencol = nextscreencol
			# move on to next tex of pixels

		#row = 0
		#self.lockpixels()
		#while row < height and row < len(itemnames):
		#	item = dataset.load(itemnames[row])
		#	if item.length == 0:
		#		itemnames = itemnames[:row] + itemnames[row + 1:]
		#		continue
		#	startrow = row
		#	col = 0
		#	while col < self.tex_width:
		#		pxoff = 0
		#		texcol = 0
		#		while texcol < len(item):
		#			c64data = item.read(offset = texcol, size = self.tex_width)
		#	#pxoff = self.pitch * row
		#	#pxdata = self.pixels[pxoff:pxoff+len(c64data)]
		#	#dstwidth = int(len(c64data) * width / maxlength + 0.5)
		#			tail = texcol + len(c64data)
		#			self.pixels[pxoff + texcol:pxoff + tail] = c64data.real * 0xff + c64data.imag * 0xff00
		#			texcol = tail
		#		pxoff += self.pitch
		#		row += 1
		#	self.render(0, startrow, None, self.tex_height) 
		#	# scale
		#	# scaling basically means writing to another texture, and then drawing that texture.
		#	# but we can draw straight to the renderer

		#	# data is c64
			

import os
class dir_dataset:
	def __init__(self, name):
		self.name = name
		self.path = os.path.join(os.path.abspath('.'), name)
		os.makedirs(self.path, exist_ok=True)
		self.maxlength = 0
		for item in self.items():
			self.maxlength = max(self.maxlength, item.length)
	def create(self, itemname, **metadata):
		item = dir_dataset.item(self, itemname)
		item.metadata = metadata
		return item
	def load(self, itemname):
		return dir_dataset.item(self, itemname)
	def list(self):
		return [fn[:-4] for fn in os.listdir(self.path) if fn.endswith('.c64')]
	def items(self):
		for item in self.list():
			yield self.load(item)
	class item:
		def __init__(self, dataset, name):
			self.dataset = dataset
			self.name = name
			self.prefix = os.path.join(self.dataset.path, self.name)
			self.bin_path = self.prefix + '.c64'
			self.metadata_path = self.prefix + '.json'
			try:
				self.bin_file = open(self.bin_path, 'x+b')
			except FileExistsError:
				# feel free to change to r+b to append
				self.bin_file = open(self.bin_path, 'rb')
			self.size_bytes = self.bin_file.seek(0, 2)
			self.bin_file.seek(0, 0)
		@property
		def metadata(self):
			with open(self.metadata_path, 'rt') as metadata_file:
				return json.load(metadata_file)
		@metadata.setter
		def metadata(self, metadata):
			assert not os.path.exists(self.metadata_path)
			with open(self.metadata_path, 'wt') as metadata_file:
				json.dump(metadata, metadata_file)
		@property
		def length(self):
			return self.size_bytes // 8
		def write(self, data):
			assert self.bin_file.tell() == self.size_bytes
			assert data.dtype == np.complex64
			self.bin_file.write(data.data)
			self.size_bytes += len(data.data)
			self.dataset.maxlength = max(self.length, self.dataset.maxlength)
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
		self.regions = set()
	# instead of start/stop being per-window
	# just run start, and call functions
	# to mark a section and complete the section
	def start(self, gui = None, test = False):
		self.source.test_mode = test
		self.running = True
		#metadata = self.source.metadata
		#dataitem = self.dataset.create(f'{self.source.channel}-{self.source.frequency}-{time.time()}', **metadata)
		last_num = 0
		with self.source as recv:
			while self.running:
				#print('WARNING: expecting dropped data in undesigned recv loop')
				data = recv()
				for region in self.regions:
					region.write(data)
				if gui is not None:
					gui.draw_dataset(dataset)
					gui.update()
				if test:
					data = data.view(dtype=np.float32)
					data = (data * 128 + 127.4).astype(int)
					dropped = (data[1:] - data[:-1]).max() - 1
					print(dropped, 'dropped mid-chunk')
					print(data[0] - last_num - 1, 'dropped inter-chunk')
					last_num = data[-1]
	def stop(self):
		self.running = False
	def start_region(self, name):
		assert self.running
		metadata = self.source.metadata
		dataitem = self.dataset.create(f'{name}-{self.source.channel}-{self.source.frequency}-{time.time()}', **metadata)
		#dataitem = self.dataset.create(f'{self.source.channel}-{self.frequency}-{time.time()}', **metadata)
		self.regions.add(dataitem)
		return dataitem
	def finish_region(self, region):
		self.regions.remove(region)
import signal
class sigeventpair:
	def __init__(self, loop, name, start_signum = signal.SIGUSR1, stop_signum = signal.SIGUSR2):
		self.start_signum = start_signum
		self.stop_signum = stop_signum
		self.loop = loop
		self.name = name
		self.region = None
	def __enter__(self):
		self._old_start = signal.signal(self.start_signum, self.on_start)
		self._old_stop = signal.signal(self.stop_signum, self.on_stop)
	def __exit__(self, *params):
		signal.signal(self.start_signum, self._old_start)
		signal.signal(self.stop_signum, self._old_stop)
	def on_start(self, signum, frame):
		if self.region is not None:
			raise Exception('double start signal')
		self.region = self.loop.start_region(self.name)
	def on_stop(self, signum, frame):
		if self.region is None:
			raise Exception('unexpected end signal')
		self.loop.finish_region(self.region)
		self.region = None
		

if __name__ == '__main__':
	#ch = display()
	#while True:
	#	with ch:
	#		ch.pixels[16] = 0x00FF00
	#	ch.update()
	scop = scopish('driver=rtlsdr,testmode=true', rate=2048000, bufsize=1024*1024*64)
	dataset = dir_dataset('test')
	gui = display()

	loop = Loop(scop, dataset)
	sigs = sigeventpair(loop, 'test')
	with sigs:
		loop.start(gui, test = False)#True)

	#scop.test_mode = True
	#last_num = 0
	#with scop as recv:
	#	while True:
	#		data = recv().view(dtype=np.float32)
	#		data = (data * 128 + 127.4).astype(int)
	#		dropped = (data[1:] - data[:-1]).max() - 1
	#		print(dropped, 'dropped mid-chunk')
	#		print(data[0] - last_num - 1, 'dropped inter-chunk')
	#		last_num = data[-1]

	#		val = data.mean()
	#		if val != last:
	#			print(val)
	#			last = val
	#		#if data.max() == 0:
	#		#	scop.gain += 1
